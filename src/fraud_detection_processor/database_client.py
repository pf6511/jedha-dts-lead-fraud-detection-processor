import uuid
from datetime import datetime
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from fraud_detection_database.models.transaction import Transaction
from fraud_detection_database.models.model_version import ModelVersion
from fraud_detection_database.models.prediction import Prediction


class DatabaseClient:

    def get_card_history(
        self,
        session: Session,
        cc_num: str,
        before: datetime,
    ) -> pd.DataFrame:
        
        statement = (
            select(Transaction)
            .where(Transaction.cc_num == cc_num, Transaction.trans_date_trans_time < before )
        ).order_by(Transaction.trans_date_trans_time.asc())

        cc_num_transactions = session.scalars(statement).all()
        return pd.DataFrame(
            [
                {
                    Transaction.cc_num.key: transaction.cc_num,
                    Transaction.amt.key: float(transaction.amt),
                    Transaction.is_fraud.key: transaction.is_fraud,
                    Transaction.trans_date_trans_time.key: (
                        transaction.trans_date_trans_time
                    ),
                }
                for transaction in cc_num_transactions
            ]
        )

    def get_model_version(
        self,
        session:Session,
        model_name: str,
        model_version: str,
    ) -> ModelVersion:
            
        statement = (
            select(ModelVersion)
            .where(
                ModelVersion.model_name == model_name,
                ModelVersion.model_version == model_version,
            )
        )

        model = session.scalar(statement)

        if model is None:
            raise ValueError(
                f"Model version not found: "
                f"{model_name} version {model_version}"
            )

        return model


    def insert_transaction(
        self,
        session:Session,
        transaction: Transaction,
    ) -> uuid.UUID:
        
        session.add(transaction)
        session.flush()
        return transaction.transaction_id


    def insert_prediction(
        self,
        session:Session,
        prediction: Prediction,
    ) -> uuid.UUID:
        
        session.add(prediction)
        session.flush()
        return prediction.prediction_id