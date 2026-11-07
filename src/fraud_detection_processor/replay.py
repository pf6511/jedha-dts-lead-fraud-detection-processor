"""Replay of the failed payments (table failed_payments).

    uv run fraud-detection-replay

Each pending failure goes through the same steps as a new payment
(process_minimised_payment). On success, its status becomes "replayed"; on a new
failure, its number of attempts increases and it stays "pending".
"""

import logging


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Imported here: secrets.env is loaded when the command runs
    from fraud_detection_processor.config import DATABASE_URL
    from fraud_detection_processor.database_client import DatabaseClient
    from fraud_detection_processor.mlflow_client import get_model_parameters
    from fraud_detection_processor.payment_preparation import payload_to_payment
    from fraud_detection_processor.workflow_processor import process_minimised_payment

    database_client = DatabaseClient(DATABASE_URL)
    model_parameters = get_model_parameters()

    with database_client.SessionLocal() as session:
        failures = database_client.get_pending_failures(session)

    logger.info("%d failed payments to replay", len(failures))

    for failure in failures:
        logger.info(
            "Replaying payment %s (failed %d time(s), last at step %s)",
            failure.source_trans_num,
            failure.attempts,
            failure.failed_step,
        )

        outcome = process_minimised_payment(
            payload_to_payment(failure.payload),
            database_client,
            model_parameters,
        )

        # Already stored in the meantime: nothing more to replay
        if outcome == "duplicate":
            with database_client.SessionLocal() as session:
                database_client.mark_pending_failure_replayed(session, failure.source_trans_num)
                session.commit()

        logger.info("Payment %s: %s", failure.source_trans_num, outcome)


if __name__ == "__main__":
    main()
