import uuid
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from fraud_detection_database.models import ModelVersion, Prediction, Transaction


class DatabaseClient:
    """Database access of the processor.

    The engine is created once for the worker's lifetime, with the
    fraud_detection_app role only (SELECT, INSERT): the processor never
    needs the admin role.
    """

    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
        )

    def get_card_history(
        self,
        session: Session,
        card_id: str,
        before: datetime,
    ) -> pd.DataFrame:
        """Previous transactions of the card (by pseudonym), before the payment."""
        statement = (
            select(Transaction)
            .where(
                Transaction.card_id == card_id,
                Transaction.trans_date_trans_time < before,
            )
            .order_by(Transaction.trans_date_trans_time.asc())
        )

        columns = [
            Transaction.card_id.key,
            Transaction.amt.key,
            Transaction.is_fraud.key,
            Transaction.trans_date_trans_time.key,
        ]

        return pd.DataFrame(
            [
                {
                    Transaction.card_id.key: transaction.card_id,
                    Transaction.amt.key: float(transaction.amt),
                    Transaction.is_fraud.key: transaction.is_fraud,
                    Transaction.trans_date_trans_time.key: transaction.trans_date_trans_time,
                }
                for transaction in session.scalars(statement).all()
            ],
            columns=columns,
        )

    def get_model_version(
        self,
        session: Session,
        model_name: str,
        model_version: str,
    ) -> ModelVersion:
        statement = select(ModelVersion).where(
            ModelVersion.model_name == model_name,
            ModelVersion.model_version == model_version,
        )

        model = session.scalar(statement)

        if model is None:
            raise ValueError(
                f"Model version not found in model_versions: "
                f"{model_name} version {model_version}"
            )

        return model

    def insert_transaction(self, session: Session, transaction: Transaction) -> uuid.UUID:
        session.add(transaction)
        session.flush()
        return transaction.transaction_id

    def insert_prediction(self, session: Session, prediction: Prediction) -> uuid.UUID:
        session.add(prediction)
        session.flush()
        return prediction.prediction_id
