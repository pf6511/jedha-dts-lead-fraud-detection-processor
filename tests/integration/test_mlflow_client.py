import pytest

from fraud_detection_contracts.inference import FEATURE_SET

from fraud_detection_processor.config import MODEL_NAME
from fraud_detection_processor.mlflow_client import get_model_parameters

pytestmark = pytest.mark.integration


def test_get_model_parameters():
    model_parameters = get_model_parameters()

    assert model_parameters.model_name == MODEL_NAME
    assert model_parameters.model_version
    assert model_parameters.run_id
    assert model_parameters.feature_set == FEATURE_SET
    assert 0 < model_parameters.p0 < 1
    assert model_parameters.alpha > 0
