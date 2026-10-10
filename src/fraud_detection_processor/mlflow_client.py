from dataclasses import dataclass

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from fraud_detection_processor.config import (
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    MODEL_NAME,
)

@dataclass(frozen=True)
class ModelParameters:
    model_name: str
    model_version: str
    run_id: str
    p0: float
    alpha: float

def get_model_parameters() -> ModelParameters:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    model_version: ModelVersion = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS,
    )

    run = client.get_run(str(model_version.run_id))

    params = run.data.params

    return ModelParameters(
        model_name=MODEL_NAME,
        model_version=model_version.version,
        run_id=str(model_version.run_id),
        p0=float(params["p0"]),
        alpha=float(params["alpha"]),
    )