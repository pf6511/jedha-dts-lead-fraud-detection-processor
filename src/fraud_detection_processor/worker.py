
import logging
import time

from fraud_detection_processor.config import WORKER_INTERVAL_SECONDS
from fraud_detection_processor.workflow_processor import process_payment

logger = logging.getLogger(__name__)


def run_worker() -> None:
    logger.info(
        "Starting fraud-detection-processor worker (interval=%s seconds)",
        WORKER_INTERVAL_SECONDS,
    )

    while True:
        try:
            process_payment()
            logger.info("Payment processed successfully")
        except Exception:
            logger.exception("Payment processing failed")

        time.sleep(WORKER_INTERVAL_SECONDS)