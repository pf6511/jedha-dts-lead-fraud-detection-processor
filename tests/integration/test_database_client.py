"""Database client against the real database (fraud_detection_app role).

The test transactions are synthetic (fictitious pseudonym, no personal
data) and are rolled back: nothing is left in the database.
"""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from fraud_detection_database.models import Transaction

from fraud_detection_processor.config import DATABASE_URL
from fraud_detection_processor.database_client import DatabaseClient
from fraud_detection_processor.mlflow_client import get_model_parameters

pytestmark = pytest.mark.integration


def synthetic_transaction(card_id: str, when: datetime, is_fraud: int) -> Transaction:
    return Transaction(
        source_trans_num=f"test-{uuid.uuid4().hex}",
        trans_date_trans_time=when,
        card_id=card_id,
        pan_last4="0000",
        merchant="test_merchant",
        category="grocery_pos",
        amt=42.0,
        city_pop=1000,
        state="IL",
        job="Tester",
        is_fraud=is_fraud,
        received_at=datetime.now(timezone.utc),
    )


def test_get_card_history_by_pseudonym():
    database_client = DatabaseClient(DATABASE_URL)

    # Random pseudonym: cannot collide with a real card
    card_id = uuid.uuid4().hex + uuid.uuid4().hex
    start = datetime(2030, 1, 1, 12, 0)
    transactions = [
        synthetic_transaction(card_id, start + timedelta(hours=hours), is_fraud)
        for hours, is_fraud in [(0, 0), (1, 1), (2, 0)]
    ]

    with database_client.SessionLocal() as session:
        try:
            for transaction in transactions:
                database_client.insert_transaction(session=session, transaction=transaction)

            history = database_client.get_card_history(
                session,
                card_id=card_id,
                before=transactions[-1].trans_date_trans_time,
            )

            # Strictly before the last one
            assert len(history) == 2
            assert list(history.columns) == [
                "card_id",
                "amt",
                "is_fraud",
                "trans_date_trans_time",
            ]
            assert (history["card_id"] == card_id).all()
            assert history["trans_date_trans_time"].is_monotonic_increasing
            assert history["is_fraud"].sum() == 1

        finally:
            session.rollback()


def test_unknown_card_has_empty_history():
    database_client = DatabaseClient(DATABASE_URL)

    with database_client.SessionLocal() as session:
        history = database_client.get_card_history(
            session,
            card_id="0" * 64,
            before=datetime(2030, 1, 1),
        )

    assert history.empty
    assert "is_fraud" in history.columns


def test_production_model_is_registered_in_database():
    """The model carrying the alias must exist in model_versions."""
    model_parameters = get_model_parameters()
    database_client = DatabaseClient(DATABASE_URL)

    with database_client.SessionLocal() as session:
        model = database_client.get_model_version(
            session=session,
            model_name=model_parameters.model_name,
            model_version=model_parameters.model_version,
        )

    assert model.model_name == model_parameters.model_name
    assert model.model_version == model_parameters.model_version
