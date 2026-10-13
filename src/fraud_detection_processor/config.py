from fraud_detection_database.environment import Environment


environment = Environment("./secrets.env")


PAYMENT_API_URL = environment.env_var("PAYMENT_API_URL")
INFERENCE_API_URL = environment.env_var("INFERENCE_API_URL")

MLFLOW_TRACKING_URI = environment.env_var("MLFLOW_TRACKING_URI")
MODEL_NAME = environment.env_var("MODEL_NAME")
MODEL_ALIAS = environment.env_var("MODEL_ALIAS")

WORKER_INTERVAL_SECONDS = int(
    environment.env_var("WORKER_INTERVAL_SECONDS")
)

DISCORD_FRAUD_WEBHOOK_URL = environment.env_var("DISCORD_FRAUD_WEBHOOK_URL")