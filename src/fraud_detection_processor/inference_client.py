import logging
import time

import httpx

from fraud_detection_contracts.inference import (
    FraudDetectionInferenceInput,
    FraudDetectionInferenceOutput,
)

from fraud_detection_processor.config import INFERENCE_API_KEY, INFERENCE_API_URL

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
WAIT_SECONDS = 2


def predict(payload: FraudDetectionInferenceInput) -> FraudDetectionInferenceOutput:
    """Call the /predict endpoint of the inference API.

    A timeout or a server error (5xx) is often temporary (Space waking
    up, overloaded): the call is retried up to 3 times. Other errors
    (401 wrong key, 422 invalid payload) would fail again: raised at once.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = httpx.post(
                INFERENCE_API_URL,
                json=payload.model_dump(),
                headers={"X-API-Key": INFERENCE_API_KEY},
                timeout=10.0,
            )
            response.raise_for_status()
            return FraudDetectionInferenceOutput.model_validate(response.json())

        except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            temporary = (
                isinstance(exc, httpx.TimeoutException)
                or exc.response.status_code >= 500
            )
            if not temporary or attempt == MAX_ATTEMPTS:
                raise

            logger.warning(
                "Inference API call failed (%s), attempt %d/%d, new try in %d s",
                type(exc).__name__,
                attempt,
                MAX_ATTEMPTS,
                WAIT_SECONDS,
            )
            time.sleep(WAIT_SECONDS)
