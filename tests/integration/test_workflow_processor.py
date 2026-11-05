"""Full cycle against all the real services.

This test processes one real payment: it calls the payment API once,
writes one transaction and one prediction to the database (kept, like
any processed payment), and may send a Discord alert.
"""

import re

import pytest
from sqlalchemy import func, select

from fraud_detection_contracts.inference import FEATURE_SET
from fraud_detection_database.models import ModelVersion, Prediction, Transaction

from fraud_detection_processor.config import DATABASE_URL
from fraud_detection_processor.database_client import DatabaseClient
from fraud_detection_processor.mlflow_client import get_model_parameters
from fraud_detection_processor.workflow_processor import process_payment

pytestmark = pytest.mark.integration


def count(session, model) -> int:
    return session.execute(select(func.count()).select_from(model)).scalar_one()


def test_process_payment():
    database_client = DatabaseClient(DATABASE_URL)

    # Arrange
    with database_client.SessionLocal() as session:
        transaction_count_before = count(session, Transaction)
        prediction_count_before = count(session, Prediction)

    # Act
    outcome = process_payment(database_client, get_model_parameters())

    # Another outcome (quarantined, duplicate, failed) is a legitimate
    # behaviour, but not the one this test checks
    assert outcome == "processed", f"cycle outcome: {outcome}"

    # Assert
    with database_client.SessionLocal() as session:
        assert count(session, Transaction) == transaction_count_before + 1
        assert count(session, Prediction) == prediction_count_before + 1

        transaction = session.scalar(
            select(Transaction).order_by(
                Transaction.received_at.desc(),
                Transaction.transaction_id.desc(),
            )
        )
        assert transaction is not None

        prediction = session.scalar(
            select(Prediction).where(Prediction.transaction_id == transaction.transaction_id)
        )
        assert prediction is not None

        model_version = session.scalar(
            select(ModelVersion).where(
                ModelVersion.model_version_id == prediction.model_version_id
            )
        )
        assert model_version is not None

    # Personal data: only the pseudonym and the last four digits are stored
    assert re.fullmatch(r"[0-9a-f]{64}", transaction.card_id)
    assert re.fullmatch(r"[0-9]{4}", transaction.pan_last4)
    for column in ("cc_num", "first", "last", "street", "dob", "gender", "lat", "long"):
        assert not hasattr(transaction, column)

    # Transaction / prediction consistency
    assert transaction.source_trans_num
    assert transaction.trans_date_trans_time is not None
    assert prediction.ground_truth == transaction.is_fraud

    # Prediction validity
    assert 0.0 <= prediction.fraud_probability <= 1.0
    assert isinstance(prediction.fraud_prediction, bool)
    assert prediction.prediction_timestamp is not None
    assert prediction.card_transaction_count >= 0

    # Prediction / model consistency
    assert model_version.feature_version == FEATURE_SET
