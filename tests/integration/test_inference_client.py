"""Inference client against the deployed inference API."""

import httpx
import pytest

from fraud_detection_contracts.inference import (
    FEATURE_SET,
    CardFeatures,
    FraudDetectionInferenceInput,
    Transaction,
)

from fraud_detection_processor.config import INFERENCE_API_URL
from fraud_detection_processor.inference_client import predict
from fraud_detection_processor.mlflow_client import get_model_parameters

pytestmark = pytest.mark.integration


def payload() -> FraudDetectionInferenceInput:
    return FraudDetectionInferenceInput(
        transaction=Transaction(
            amt=258.37,
            city_pop=725,
            merchant="fraud_Skiles LLC",
            category="home",
            state="IL",
            job="Environmental consultant",
        ),
        card_features=CardFeatures(
            distance_km=20.0,
            card_transaction_count=10,
            card_fraud_count=1,
            card_fraud_rate=0.1,
        ),
    )


def test_predict_returns_production_model_version():
    result = predict(payload())

    assert 0.0 <= result.fraud_probability <= 1.0
    assert isinstance(result.fraud_prediction, bool)
    assert result.feature_set == FEATURE_SET

    # The API serves the model carrying the alias read by the processor
    assert result.model_version == get_model_parameters().model_version


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong-key"}])
def test_api_rejects_missing_or_wrong_key(headers):
    response = httpx.post(
        INFERENCE_API_URL,
        json=payload().model_dump(),
        headers=headers,
        timeout=10.0,
    )
    assert response.status_code == 401
