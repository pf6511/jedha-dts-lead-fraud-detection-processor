"""Processing of one payment, from the payment API to the database.

Quality control and error correction:
- an invalid payment is quarantined (quarantined_payments), not processed;
- a payment already stored is ignored (deduplication);
- a valid payment whose processing fails is stored in failed_payments
  and can be replayed later (replay.py).

Personal data stops at the minimisation (step 3): everything after
works on the minimised payment. Logs never contain the card number.

The parameters of the production model (p0, alpha, version) are read
once from MLflow when the worker starts and passed to each cycle: after
a change of the production alias, restart the inference API and the
worker.
"""

import logging
import uuid
from datetime import datetime, timezone

from fraud_detection_contracts.inference import (
    FraudDetectionInferenceInput,
    FraudDetectionInferenceOutput,
)
from fraud_detection_database.models import Prediction, QuarantinedPayment, Transaction
from fraud_detection_features.features import build_inference_features

from fraud_detection_processor.config import CARD_ID_KEY
from fraud_detection_processor.database_client import DatabaseClient
from fraud_detection_processor.inference_client import predict
from fraud_detection_processor.mlflow_client import ModelParameters
from fraud_detection_processor.notification_client import notify_fraud_detected
from fraud_detection_processor.payment_client import get_payment
from fraud_detection_processor.payment_preparation import (
    PreparedPayment,
    build_inference_input,
    payment_to_payload,
    prepare_payment,
)
from fraud_detection_processor.validation import quarantine_payload, validate_payment

logger = logging.getLogger(__name__)

# =====================================================================
# full process of payment
# =====================================================================

def process_payment(
    database_client: DatabaseClient,
    model_parameters: ModelParameters,
) -> str:
    """Process one new payment; return the outcome: quarantined,
    duplicate, processed or failed."""

    # 1. Get the payment
    logger.info("Step 1/8 - Retrieve payment")
    payment_data = get_payment()

    # 2. Validate it; an invalid payment is quarantined
    logger.info("Step 2/8 - Validate payment")
    errors = validate_payment(payment_data)

    if errors:
        quarantine_payment(payment_data, errors, database_client)
        return "quarantined"

    # 3. Minimise: personal data stops here
    prepared = prepare_payment(payment_data, CARD_ID_KEY)
    logger.info(
        "Step 3/8 - Payment %s minimised (%d columns removed: %s)",
        trans_num_of(prepared),
        len(prepared.dropped_columns),
        ", ".join(prepared.dropped_columns),
    )
    # next steps on prepared payment
    return process_minimised_payment(prepared, database_client, model_parameters)


def quarantine_payment(
    payment_data: dict,
    errors: list[str],
    database_client: DatabaseClient,
) -> None:
    """Store an invalid payment in quarantined_payments (non-personal fields only)."""
    payload = quarantine_payload(payment_data)
    reason = "; ".join(errors)

    with database_client.SessionLocal() as session:
        database_client.insert_quarantined_payment(
            session,
            QuarantinedPayment(
                source_trans_num=payload.get("trans_num"),
                payload=payload,
                reason=reason,
            ),
        )
        session.commit()

    logger.warning("Payment %s quarantined: %s", payload.get("trans_num"), reason)


# =====================================================================
# Steps 4 to 8: minimised payment (new or replayed)
# =====================================================================

def process_minimised_payment(
    prepared: PreparedPayment,
    database_client: DatabaseClient,
    model_parameters: ModelParameters,
) -> str:
    """Steps 4 to 8, shared by new payments and replayed ones.

    From step 5, a failure stores the payment in failed_payments, with
    the step where it failed.
    """
    # 4. Deduplication
    if is_duplicate(prepared, database_client):
        return "duplicate"

    step = "features"
    try:
        # 5. Card history and features
        inference_input = build_inference_payload(prepared, database_client, model_parameters)

        # 6. Inference API
        step = "inference_api"
        inference_output = get_prediction(inference_input, model_parameters)

        # 7. Transaction and prediction in the database
        step = "persistence"
        transaction_id = store_transaction_and_prediction(
            prepared, inference_input, inference_output, database_client
        )

    except Exception as exc:
        logger.exception("Payment %s failed at step %s", trans_num_of(prepared), step)
        save_failure(prepared, step, exc, database_client)
        return "failed"

    # 8. Alert, after the commit
    send_alert_if_fraud(prepared, inference_output, transaction_id)

    return "processed"


def is_duplicate(prepared: PreparedPayment, database_client: DatabaseClient) -> bool:
    """Step 4: has this payment already been stored?"""
    trans_num = trans_num_of(prepared)

    with database_client.SessionLocal() as session:
        duplicate = database_client.transaction_exists(session, trans_num)

    if duplicate:
        logger.info("Step 4/8 - Payment %s already stored: ignored", trans_num)
    return duplicate


def build_inference_payload(
    prepared: PreparedPayment,
    database_client: DatabaseClient,
    model_parameters: ModelParameters,
) -> FraudDetectionInferenceInput:
    """Step 5: card history (strictly before the payment) and features.

    The history is read before the transaction time, so a replayed
    payment gets the same features as on its first processing.
    """
    logger.info("Step 5/8 - Load card history and build features")

    with database_client.SessionLocal() as session:
        history = database_client.get_card_history(
            session=session,
            card_id=prepared.payment.iloc[0]["card_id"],
            before=prepared.transaction_time,
        )

    features = build_inference_features(
        payment=prepared.payment,
        history=history,
        p0=model_parameters.p0,
        alpha=model_parameters.alpha,
    )
    return build_inference_input(features.iloc[0])


def get_prediction(
    inference_input: FraudDetectionInferenceInput,
    model_parameters: ModelParameters,
) -> FraudDetectionInferenceOutput:
    """Step 6: call the inference API (it retries temporary errors itself)."""
    logger.info("Step 6/8 - Call inference API")
    inference_output = predict(inference_input)

    # p0 and alpha must come from the model that predicted
    if inference_output.model_version != model_parameters.model_version:
        raise RuntimeError(
            f"Model mismatch: features built for version "
            f"{model_parameters.model_version}, predicted by version "
            f"{inference_output.model_version} "
            "(production alias changed: restart the inference API and the worker)"
        )

    logger.info(
        "Step 6/8 - Predicted by %s version %s: probability %.4f",
        inference_output.model_name,
        inference_output.model_version,
        inference_output.fraud_probability,
    )
    return inference_output


def store_transaction_and_prediction(
    prepared: PreparedPayment,
    inference_input: FraudDetectionInferenceInput,
    inference_output: FraudDetectionInferenceOutput,
    database_client: DatabaseClient,
) -> uuid.UUID:
    """Step 7: transaction + prediction in one database transaction.

    A replayed payment also closes its failure in the same commit.
    """
    logger.info("Step 7/8 - Persist transaction and prediction")
    row = prepared.payment.iloc[0]
    card_features = inference_input.card_features

    with database_client.SessionLocal() as session:
        try:
            model_version = database_client.get_model_version(
                session=session,
                model_name=inference_output.model_name,
                model_version=inference_output.model_version,
            )

            transaction_id = database_client.insert_transaction(
                session=session,
                transaction=Transaction(
                    source_trans_num=str(row["trans_num"]),
                    trans_date_trans_time=prepared.transaction_time,
                    card_id=str(row["card_id"]),
                    pan_last4=prepared.pan_last4,
                    merchant=str(row["merchant"]),
                    category=str(row["category"]),
                    amt=float(row["amt"]),
                    city_pop=int(row["city_pop"]),
                    state=str(row["state"]),
                    job=str(row["job"]),
                    is_fraud=int(row["is_fraud"]),
                    received_at=datetime.now(timezone.utc),
                ),
            )

            database_client.insert_prediction(
                session=session,
                prediction=Prediction(
                    transaction_id=transaction_id,
                    model_version_id=model_version.model_version_id,
                    fraud_probability=inference_output.fraud_probability,
                    fraud_prediction=inference_output.fraud_prediction,
                    ground_truth=int(row["is_fraud"]),
                    # Snapshot of the features actually sent to the model
                    distance_km=card_features.distance_km,
                    card_transaction_count=card_features.card_transaction_count,
                    card_fraud_count=card_features.card_fraud_count,
                    card_fraud_rate=card_features.card_fraud_rate,
                ),
            )

            database_client.mark_pending_failure_replayed(session, str(row["trans_num"]))

            session.commit()

        except Exception:
            session.rollback()
            raise

    logger.info("Step 7/8 - Persisted, transaction id %s", transaction_id)
    return transaction_id


def send_alert_if_fraud(
    prepared: PreparedPayment,
    inference_output: FraudDetectionInferenceOutput,
    transaction_id: uuid.UUID,
) -> None:
    """Step 8: Discord alert when fraud is predicted (masked card only)."""
    if not inference_output.fraud_prediction:
        logger.info("Step 8/8 - No fraud detected, no alert")
        return

    sent = notify_fraud_detected(
        transaction_id=str(transaction_id),
        pan_last4=prepared.pan_last4,
        amount=float(prepared.payment.iloc[0]["amt"]),
        fraud_probability=inference_output.fraud_probability,
        model_version=inference_output.model_version,
    )
    logger.info("Step 8/8 - Fraud detected, alert sent: %s", sent)


# =====================================================================
# Helpers
# =====================================================================

def save_failure(
    prepared: PreparedPayment,
    step: str,
    exc: Exception,
    database_client: DatabaseClient,
) -> None:
    """Store the failed payment to replay it later."""
    trans_num = trans_num_of(prepared)

    try:
        with database_client.SessionLocal() as session:
            database_client.save_failure(
                session,
                source_trans_num=trans_num,
                payload=payment_to_payload(prepared),
                failed_step=step,
                error=f"{type(exc).__name__}: {exc}"[:500],
            )
            session.commit()
        logger.warning("Payment %s stored in failed_payments", trans_num)

    except Exception:
        # The database itself is down: the log is the only trace
        logger.exception("Payment %s could not be stored in failed_payments", trans_num)


def trans_num_of(prepared: PreparedPayment) -> str:
    return str(prepared.payment.iloc[0]["trans_num"])
