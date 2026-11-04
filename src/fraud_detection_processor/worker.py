import logging
import time

from fraud_detection_processor.config import DATABASE_URL, WORKER_INTERVAL_SECONDS
from fraud_detection_processor.database_client import DatabaseClient
from fraud_detection_processor.mlflow_client import get_model_parameters
from fraud_detection_processor.workflow_processor import process_payment

logger = logging.getLogger(__name__)


def run_worker() -> None:
    logger.info(
        "Starting fraud-detection-processor worker (interval=%s seconds)",
        WORKER_INTERVAL_SECONDS,
    )

    # Fail fast: an incompatible production model stops the worker
    # at startup instead of failing on every payment.
    model_parameters = get_model_parameters()
    logger.info(
        "Production model: %s version %s (feature set %s)",
        model_parameters.model_name,
        model_parameters.model_version,
        model_parameters.feature_set,
    )

    database_client = DatabaseClient(DATABASE_URL)

    while True:
        try:
            process_payment(database_client)
            logger.info("Payment processed successfully")
        except Exception:
            logger.exception("Payment processing failed")

        time.sleep(WORKER_INTERVAL_SECONDS)
