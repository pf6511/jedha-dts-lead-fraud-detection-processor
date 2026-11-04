from dataclasses import dataclass

import mlflow
from mlflow import MlflowClient

from fraud_detection_contracts.inference import FEATURE_SET

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
    feature_set: str
    p0: float
    alpha: float


def get_model_parameters() -> ModelParameters:
    """Read the feature parameters of the model that contains the alias.

    p0 and alpha must be those of the model that scores the payment:
    the card fraud rate is smoothed with them.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    params = client.get_run(str(model_version.run_id)).data.params
    # check feature_set name match FEATURE8SET
    feature_set = params.get("feature_set")
    if feature_set != FEATURE_SET:
        raise RuntimeError(
            f"Model {MODEL_NAME} version {model_version.version} uses feature "
            f"set {feature_set!r}; this processor builds {FEATURE_SET!r}"
        )

    return ModelParameters(
        model_name=MODEL_NAME,
        model_version=str(model_version.version),
        run_id=str(model_version.run_id),
        feature_set=feature_set,
        p0=float(params["p0"]),
        alpha=float(params["alpha"]),
    )
