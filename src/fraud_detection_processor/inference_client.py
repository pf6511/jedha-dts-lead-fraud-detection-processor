import httpx

from fraud_detection_contracts.inference import (
    FraudDetectionInferenceInput,
    FraudDetectionInferenceOutput,
)

from fraud_detection_processor.config import INFERENCE_API_KEY, INFERENCE_API_URL


def predict(payload: FraudDetectionInferenceInput) -> FraudDetectionInferenceOutput:
    """Call the /predict endpoint of the inference API (URL of the endpoint)."""
    response = httpx.post(
        INFERENCE_API_URL,
        json=payload.model_dump(),
        headers={"X-API-Key": INFERENCE_API_KEY},
        timeout=10.0,
    )
    response.raise_for_status()

    return FraudDetectionInferenceOutput.model_validate(response.json())
