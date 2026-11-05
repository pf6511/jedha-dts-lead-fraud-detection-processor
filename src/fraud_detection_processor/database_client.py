import uuid
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from fraud_detection_database.models import (
    FailedPayment,
    ModelVersion,
    Prediction,
    QuarantinedPayment,
    Transaction,
)


class DatabaseClient:
    """Database access of the processor.

    The engine is created once for the worker's lifetime, with the
    fraud_detection_app role only: the processor never needs the admin
    role.
    """

    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
        )

    # -----------------------------------------------------------------
    # Transactions and predictions
    # -----------------------------------------------------------------

    def transaction_exists(self, session: Session, source_trans_num: str) -> bool:
        """Deduplication: has this payment already been stored?"""
        statement = select(Transaction.transaction_id).where(
            Transaction.source_trans_num == source_trans_num
        )
        return session.scalar(statement) is not None

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

    # -----------------------------------------------------------------
    # Quarantine and failed payments
    # -----------------------------------------------------------------

    def insert_quarantined_payment(self, session: Session, quarantined: QuarantinedPayment) -> None:
        session.add(quarantined)

    def save_failure(
        self,
        session: Session,
        source_trans_num: str,
        payload: dict,
        failed_step: str,
        error: str,
    ) -> None:
        """Store a failed payment; if it already failed, count one more attempt."""
        failure = session.scalar(
            select(FailedPayment).where(FailedPayment.source_trans_num == source_trans_num)
        )

        if failure is None:
            session.add(
                FailedPayment(
                    source_trans_num=source_trans_num,
                    payload=payload,
                    failed_step=failed_step,
                    error=error,
                )
            )
        else:
            failure.attempts += 1
            failure.failed_step = failed_step
            failure.error = error
            failure.failed_at = datetime.now(timezone.utc)

    def get_pending_failures(self, session: Session) -> list[FailedPayment]:
        statement = (
            select(FailedPayment)
            .where(FailedPayment.status == "pending")
            .order_by(FailedPayment.failed_at.asc())
        )
        return list(session.scalars(statement).all())

    def mark_replayed(self, session: Session, source_trans_num: str) -> None:
        """Close the failure of a payment, if there is one."""
        failure = session.scalar(
            select(FailedPayment).where(
                FailedPayment.source_trans_num == source_trans_num,
                FailedPayment.status == "pending",
            )
        )
        if failure is not None:
            failure.status = "replayed"
            failure.replayed_at = datetime.now(timezone.utc)
