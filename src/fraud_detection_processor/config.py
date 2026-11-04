"""Processor configuration, read once from secrets.env / the environment.

Secrets (never committed, see secrets.env.example):
- CARD_ID_HMAC_KEY       key of the card number pseudonymisation; must be
                         the same key as the one used to prepare the
                         training dataset, otherwise card histories do
                         not match
- INFERENCE_API_KEY      sent in the X-API-Key header
- MLFLOW_TRACKING_TOKEN  read directly by the MLflow client
- DATABASE_URL           connection with the fraud_detection_app role
                         (SELECT, INSERT only)
"""

from fraud_detection_database.environment import Environment
from fraud_detection_features.privacy import load_card_id_key

environment = Environment("./secrets.env")


PAYMENT_API_URL = environment.env_var("PAYMENT_API_URL")

INFERENCE_API_URL = environment.env_var("INFERENCE_API_URL")
INFERENCE_API_KEY = environment.env_var("INFERENCE_API_KEY")

MLFLOW_TRACKING_URI = environment.env_var("MLFLOW_TRACKING_URI")
MODEL_NAME = environment.env_var("MODEL_NAME")
MODEL_ALIAS = environment.env_var("MODEL_ALIAS")

DATABASE_URL = environment.env_var("DATABASE_URL")

WORKER_INTERVAL_SECONDS = int(environment.env_var("WORKER_INTERVAL_SECONDS"))

DISCORD_FRAUD_WEBHOOK_URL = environment.env_var("DISCORD_FRAUD_WEBHOOK_URL")

# Loaded at startup: the worker does not start without a valid key
CARD_ID_KEY = load_card_id_key()
