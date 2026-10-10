from fraud_detection_processor.config import (
    MODEL_ALIAS,
    MODEL_NAME,
)
from fraud_detection_processor.mlflow_client import get_model_parameters


def test_get_model_parameters():
    model_parameters = get_model_parameters()

    assert model_parameters.model_name == MODEL_NAME
    assert model_parameters.model_version
    assert model_parameters.run_id
    assert model_parameters.p0 >= 0
    assert model_parameters.alpha > 0