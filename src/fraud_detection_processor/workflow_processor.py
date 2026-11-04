"""Processing of one payment, from the payment API to the database.

Personal data stops at step 1: everything after works on the minimised
payment (card_id, no name, address, date of birth, gender or
coordinates). Logs never contain the card number.
"""

import logging
from datetime import datetime, timezone

from fraud_detection_database.models import Prediction, Transaction
from fraud_detection_features.features import build_inference_features

from fraud_detection_processor.config import CARD_ID_KEY
from fraud_detection_processor.database_client import DatabaseClient
from fraud_detection_processor.inference_client import predict
from fraud_detection_processor.mlflow_client import get_model_parameters
from fraud_detection_processor.notification_client import notify_fraud_detected
from fraud_detection_processor.payment_client import get_payment
from fraud_detection_processor.payment_preparation import (
    build_inference_input,
    prepare_payment,
)

logger = logging.getLogger(__name__)


def process_payment(database_client: DatabaseClient) -> None:

    # ------------------------------------------------------------------
    # 1. Get the payment and minimise it immediately
    # ------------------------------------------------------------------
    logger.info("Step 1/8 - Retrieve payment and minimise personal data")

    prepared_payment = prepare_payment(get_payment(), CARD_ID_KEY)
    row = prepared_payment.payment.iloc[0]

    logger.info(
        "Step 1/8 - Payment %s minimised (%d columns removed: %s)",
        row["trans_num"],
        len(prepared_payment.dropped_columns),
        ", ".join(prepared_payment.dropped_columns),
    )

    # ------------------------------------------------------------------
    # 2. Parameters of the production model (MLflow)
    # ------------------------------------------------------------------
    logger.info("Step 2/8 - Load model parameters from MLflow")

    model_parameters = get_model_parameters()

    # ------------------------------------------------------------------
    # 3. Card history, by pseudonym, strictly before the payment
    # ------------------------------------------------------------------
    logger.info("Step 3/8 - Load card history")

    with database_client.SessionLocal() as session:
        history = database_client.get_card_history(
            session=session,
            card_id=row["card_id"],
            before=prepared_payment.transaction_time,
        )

    logger.info("Step 3/8 - Card history loaded: %d transactions", len(history))

    # ------------------------------------------------------------------
    # 4. Features and API payload
    # ------------------------------------------------------------------
    logger.info("Step 4/8 - Build inference features and payload")

    features = build_inference_features(
        payment=prepared_payment.payment,
        history=history,
        p0=model_parameters.p0,
        alpha=model_parameters.alpha,
    )
    features_row = features.iloc[0]

    inference_input = build_inference_input(features_row)

    # ------------------------------------------------------------------
    # 5. Inference API
    # ------------------------------------------------------------------
    logger.info("Step 5/8 - Call inference API")

    inference_output = predict(inference_input)

    # p0 and alpha must come from the model that actually scored:
    # if the alias moved between steps 2 and 5, the payment is rejected.
    if inference_output.model_version != model_parameters.model_version:
        raise RuntimeError(
            f"Model mismatch: features built for version {model_parameters.model_version}, "
            f"predicted by version {inference_output.model_version} "
            "(alias moved; restart the inference API)"
        )

    logger.info(
        "Step 5/8 - Predicted by %s version %s: probability %.4f",
        inference_output.model_name,
        inference_output.model_version,
        inference_output.fraud_probability,
    )

    # ------------------------------------------------------------------
    # 6. Model version registered in the database
    # ------------------------------------------------------------------
    logger.info("Step 6/8 - Get model version metadata from DB")

    with database_client.SessionLocal() as session:
        model_version = database_client.get_model_version(
            session=session,
            model_name=inference_output.model_name,
            model_version=inference_output.model_version,
        )

    # ------------------------------------------------------------------
    # 7. Persist transaction + prediction atomically
    # ------------------------------------------------------------------
    logger.info("Step 7/8 - Persist transaction and prediction")

    transaction = Transaction(
        source_trans_num=str(row["trans_num"]),
        trans_date_trans_time=prepared_payment.transaction_time,
        card_id=str(row["card_id"]),
        pan_last4=prepared_payment.pan_last4,
        merchant=str(row["merchant"]),
        category=str(row["category"]),
        amt=float(row["amt"]),
        city_pop=int(row["city_pop"]),
        state=str(row["state"]),
        job=str(row["job"]),
        is_fraud=int(row["is_fraud"]),
        received_at=datetime.now(timezone.utc),
    )

    with database_client.SessionLocal() as session:
        try:
            transaction_id = database_client.insert_transaction(
                session=session,
                transaction=transaction,
            )

            card_features = inference_input.card_features

            prediction = Prediction(
                transaction_id=transaction_id,
                model_version_id=model_version.model_version_id,
                fraud_probability=inference_output.fraud_probability,
                fraud_prediction=inference_output.fraud_prediction,
                ground_truth=int(row["is_fraud"]),
                # Snapshot of the features sent to inference api so to the model for prediction
                distance_km=card_features.distance_km,
                card_transaction_count=card_features.card_transaction_count,
                card_fraud_count=card_features.card_fraud_count,
                card_fraud_rate=card_features.card_fraud_rate,
            )

            database_client.insert_prediction(session=session, prediction=prediction)

            session.commit()

        except Exception:
            session.rollback()
            raise

    logger.info("Step 7/8 - Persisted, transaction id %s", transaction_id)

    # ------------------------------------------------------------------
    # 8. Discord alert, after the commit
    # ------------------------------------------------------------------
    if inference_output.fraud_prediction:
        sent = notify_fraud_detected(
            transaction_id=str(transaction_id),
            pan_last4=prepared_payment.pan_last4,
            amount=float(row["amt"]),
            fraud_probability=inference_output.fraud_probability,
            model_version=inference_output.model_version,
        )
        logger.info("Step 8/8 - Fraud detected, alert sent: %s", sent)
    else:
        logger.info("Step 8/8 - No fraud detected, no alert")
