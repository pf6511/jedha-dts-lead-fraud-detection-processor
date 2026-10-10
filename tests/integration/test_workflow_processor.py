from fraud_detection_database.database import Database
from fraud_detection_database.environment import Environment
from fraud_detection_processor.workflow_processor import process_payment

from fraud_detection_database.models.prediction import Prediction
from fraud_detection_database.models.transaction import Transaction
from fraud_detection_database.models.model_version import ModelVersion

from sqlalchemy import select, func

def get_database_access() -> Database:
    environment = Environment("./secrets.env")
    return Database(
        database_url=environment.env_var("DATABASE_URL"),
        admin_database_url=environment.env_var("ADMIN_DATABASE_URL"),
    )

def test_process_payment():
    database = get_database_access()

    # Arrange
    with database.SessionLocal() as session:
        transaction_count_before = session.execute(
            select(func.count()).select_from(Transaction)
        ).scalar_one()

        prediction_count_before = session.execute(
            select(func.count()).select_from(Prediction)
        ).scalar_one()

    # Act
    process_payment()

    # Assert
    with database.SessionLocal() as session:
        transaction_count_after = session.execute(
            select(func.count()).select_from(Transaction)
        ).scalar_one()

        prediction_count_after = session.execute(
            select(func.count()).select_from(Prediction)
        ).scalar_one()

        assert transaction_count_after == transaction_count_before + 1
        assert prediction_count_after == prediction_count_before + 1

        transaction = session.scalar(
            select(Transaction).order_by(
                Transaction.received_at.desc(),
                Transaction.transaction_id.desc(),
            )
        )

        assert transaction is not None

        prediction = session.scalar(
            select(Prediction).where(
                Prediction.transaction_id == transaction.transaction_id
            )
        )

        assert prediction is not None

        model_version = session.scalar(
            select(ModelVersion).where(
                ModelVersion.model_version_id == prediction.model_version_id
            )
        )

        assert model_version is not None

        # Transaction / prediction consistency
        assert prediction.transaction_id == transaction.transaction_id

        assert prediction.ground_truth is not None
        assert prediction.ground_truth == transaction.is_fraud

        # Prediction validity
        assert prediction.fraud_probability is not None
        assert 0.0 <= prediction.fraud_probability <= 1.0

        assert prediction.fraud_prediction is not None
        assert isinstance(prediction.fraud_prediction, bool)

        # Prediction / model consistency
        assert prediction.model_version_id == model_version.model_version_id

        # Prediction / model consistency
        assert prediction.model_version_id == model_version.model_version_id
        assert transaction.source_trans_num is not None
        assert transaction.cc_num is not None
        assert transaction.trans_date_trans_time is not None
        assert prediction.prediction_timestamp is not None