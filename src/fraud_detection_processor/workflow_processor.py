from datetime import datetime, timezone

import pandas as pd

from fraud_detection_database.models.prediction import Prediction
from fraud_detection_database.models.transaction import Transaction

from fraud_detection_features.features import build_inference_features

from fraud_detection_contracts.inference import (
    CardFeatures,
    FraudDetectionInferenceInput,
    Transaction as InferenceTransaction,
)

from fraud_detection_processor.database_client import DatabaseClient
from fraud_detection_processor.inference_client import predict
from fraud_detection_processor.mlflow_client import get_model_parameters
from fraud_detection_processor.payment_client import get_payment

from fraud_detection_processor.notification_client import (
    notify_fraud_detected,
)

import logging

logger = logging.getLogger(__name__)

def process_payment() -> None:

    from fraud_detection_database.database import Database
    from fraud_detection_database.environment import Environment
    # ------------------------------------------------------------------
    # 1. Get payment from the payment API
    # ------------------------------------------------------------------
    payment_data = get_payment()

    logger.info("Step 1.1/9 - Call payment API - Retrieving payment")

    payment = pd.DataFrame(
        payment_data["data"],
        columns=payment_data["columns"],
    )

    logger.info("Step 1.2/9 - payment retrieved")
    if len(payment) != 1:
        raise ValueError(
            f"Expected exactly one payment, got {len(payment)}"
        )

    # ------------------------------------------------------------------
    # 2. Extract current transaction timestamp
    # ------------------------------------------------------------------

    logger.info("Step 2/9 - Extract current payment timestamp (for card history)")
    current_time = pd.to_datetime(payment.iloc[0]["current_time"], unit='ms').to_pydatetime()

    # ------------------------------------------------------------------
    # 3. Get active / production model parameters from MLflow
    # ------------------------------------------------------------------
    logger.info("Step 3.1/9 - load model parameters from Mlflow")

    model_parameters = get_model_parameters()

    logger.info("Step 3.2/9 - model parameters loaded")

    # ------------------------------------------------------------------
    # 4. Get previous transactions for this card
    # ------------------------------------------------------------------

    environment = Environment("./secrets.env")

    database = Database(
        database_url=environment.env_var("DATABASE_URL"),
        admin_database_url=environment.env_var("ADMIN_DATABASE_URL"),
    )

    database_client = DatabaseClient()

    cc_num = str(payment.iloc[0]["cc_num"])

    logger.info("Step 4.1/9 - Load card paiement history from DB")
    with database.SessionLocal() as session:
        history = database_client.get_card_history(
            session=session,
            cc_num=cc_num,
            before=current_time,
        )

    logger.info("Step 4.2/9 - card paiemnt history loaded : %d", len(history))

    # ------------------------------------------------------------------
    # 5. Build inference features
    # ------------------------------------------------------------------

    logger.info("Step 5/9 - build inference features")

    inference_features = build_inference_features(
        payment=payment,
        history=history,
        p0=model_parameters.p0,
        alpha=model_parameters.alpha,
    )   

    inference_x_row = inference_features.iloc[0]

    # ------------------------------------------------------------------
    # 6. Build inference API payload
    # ------------------------------------------------------------------

    logger.info("Step 6/9 - build API payload / input")

    transaction = InferenceTransaction(
            amt = float(inference_x_row['amt']),
            lat=float(inference_x_row["lat"]),
            long=float(inference_x_row["long"]),
            city_pop=int(inference_x_row["city_pop"]),
            merch_lat=float(inference_x_row["merch_lat"]),
            merch_long=float(inference_x_row["merch_long"]),
            merchant=str(inference_x_row["merchant"]),
            category=str(inference_x_row["category"]),
            gender=str(inference_x_row["gender"]),
            state=str(inference_x_row["state"]),
            job=str(inference_x_row["job"])       
    )

    card_avg_amount = inference_x_row["card_avg_amount"]

    card_features = CardFeatures(
            distance_km=float(inference_x_row["distance_km"]),
            card_transaction_count=int(inference_x_row["card_transaction_count"]),
            card_fraud_count=int(inference_x_row["card_fraud_count"]),
            card_fraud_rate=float(inference_x_row["card_fraud_rate"]),
            card_avg_amount=(
                    None
                    if pd.isna(card_avg_amount)
                    else float(card_avg_amount)
                ),
    )

    inference_input = FraudDetectionInferenceInput(transaction=transaction, card_features=card_features)


    # ------------------------------------------------------------------
    # 7. Call inference API
    # ------------------------------------------------------------------

    logger.info("Step 7.1/9 - Call inference API")

    inference_output = predict( inference_input )

    logger.info("Step 7.2/9 - Inference Output returned")

    # ------------------------------------------------------------------
    # 8. Get database model version
    # ------------------------------------------------------------------

    logger.info("Step 8.1/9 - Get model version metadata from DB")
    with database.SessionLocal() as session:
        model_version = database_client.get_model_version(
            session=session,
            model_name=model_parameters.model_name,
            model_version=model_parameters.model_version,
        )

    logger.info("Step 8.2/9 - Model version metadata returned")

  # ------------------------------------------------------------------
    # 9. Persist transaction + prediction atomically
    # ------------------------------------------------------------------

    logger.info("Step 9.1/9 - Persist Transaction and Prediction to DB")

    transaction = Transaction(
        source_trans_num=str(inference_x_row["trans_num"]),
        trans_date_trans_time=current_time,
        cc_num=cc_num,
        merchant=str(inference_x_row["merchant"]),
        category=str(inference_x_row["category"]),
        amt=float(inference_x_row["amt"]),
        first=str(inference_x_row["first"]),
        last=str(inference_x_row["last"]),
        gender=str(inference_x_row["gender"]),
        street=str(inference_x_row["street"]),
        city=str(inference_x_row["city"]),
        state=str(inference_x_row["state"]),
        zip=int(inference_x_row["zip"]),
        lat=float(inference_x_row["lat"]),
        long=float(inference_x_row["long"]),
        city_pop=int(inference_x_row["city_pop"]),
        job=str(inference_x_row["job"]),
        dob=pd.to_datetime(inference_x_row["dob"]).date(),
        merch_lat=float(inference_x_row["merch_lat"]),
        merch_long=float(inference_x_row["merch_long"]),
        is_fraud=int(inference_x_row["is_fraud"]),
        received_at=datetime.now(timezone.utc),
    )

    with database.SessionLocal() as session:
        try:
            transaction_id = database_client.insert_transaction(
                session=session,
                transaction=transaction,
            )

            prediction = Prediction(
                transaction_id=transaction_id,
                model_version_id=model_version.model_version_id,
                fraud_probability=inference_output.fraud_probability,
                fraud_prediction=inference_output.fraud_prediction,
                ground_truth=int(inference_x_row["is_fraud"]),
                distance_km=float(inference_x_row["distance_km"]),
                card_transaction_count=int(
                    inference_x_row["card_transaction_count"]
                ),
                card_fraud_count=int(
                    inference_x_row["card_fraud_count"]
                ),
                card_fraud_rate=float(
                    inference_x_row["card_fraud_rate"]
                ),
                card_avg_amount=float(
                    inference_x_row["card_avg_amount"]
                ),
            )

            database_client.insert_prediction(
                session=session,
                prediction=prediction,
            )

            session.commit()
            logger.info("Step 9.2/9.2 - Transaction and Prediction persisted to DB")
        

        except Exception:
            session.rollback()
            raise

        # ------------------------------------------------------------------
    # 10. Notify Discord after successful database commit
    # ------------------------------------------------------------------

    if inference_output.fraud_prediction:
        notify_fraud_detected(
            transaction_id=transaction_id.int,
            amount=float(inference_x_row["amt"]),
            fraud_probability=float(
                inference_output.fraud_probability
            ),
        )
        logger.info("Fraud Detected - Notification sent to thread")