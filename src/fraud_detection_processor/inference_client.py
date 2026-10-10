import httpx

from fraud_detection_processor.config import INFERENCE_API_URL


from fraud_detection_contracts.inference import (
    FraudDetectionInferenceInput,
    FraudDetectionInferenceOutput,
)

def predict(
    payload: FraudDetectionInferenceInput,
) -> FraudDetectionInferenceOutput:

    response = httpx.post(
        f"{INFERENCE_API_URL}",
        json=payload.model_dump(),
        timeout=10.0,
    )
    response.raise_for_status()

    return FraudDetectionInferenceOutput.model_validate(
        response.json()
    )