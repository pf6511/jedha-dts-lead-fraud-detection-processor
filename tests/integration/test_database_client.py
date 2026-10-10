from sqlalchemy import select, func

from fraud_detection_database.database import Database
from fraud_detection_database.environment import Environment

from fraud_detection_database.models.transaction import Transaction

from fraud_detection_processor.database_client import DatabaseClient

from pathlib import Path
import pandas as pd
from datetime import datetime, timezone

TEST_DATA_PATH = Path("tests/data/transactions_test_with_fraud.csv")

def get_database_access() -> Database:
    environment = Environment("./secrets.env")
    return Database(
        database_url=environment.env_var("DATABASE_URL"),
        admin_database_url=environment.env_var("ADMIN_DATABASE_URL"),
    )

def test_get_card_history():
    df = pd.read_csv(TEST_DATA_PATH)
    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"]
    )
    cc_num = df["cc_num"].iloc[0]

    sample = (
        df[df["cc_num"] == cc_num]
        .sort_values("trans_date_trans_time")
        .head(3)
    )

    assert len(sample) == 3

    transactions = [
        Transaction(
            source_trans_num=str(row["trans_num"]),
            trans_date_trans_time=row["trans_date_trans_time"],
            cc_num=cc_num,
            merchant=str(row["merchant"]),
            category=str(row["category"]),
            amt=float(row["amt"]),
            first=str(row["first"]),
            last=str(row["last"]),
            gender=str(row["gender"]),
            street=str(row["street"]),
            city=str(row["city"]),
            state=str(row["state"]),
            zip=int(row["zip"]),
            lat=float(row["lat"]),
            long=float(row["long"]),
            city_pop=int(row["city_pop"]),
            job=str(row["job"]),
            dob=pd.to_datetime(row["dob"]).date(),
            merch_lat=float(row["merch_lat"]),
            merch_long=float(row["merch_long"]),
            is_fraud=int(row["is_fraud"]),
            received_at=datetime.now(timezone.utc),
        )
        for _, row in sample.iterrows()
    ]

    database_client = DatabaseClient()

    with get_database_access().SessionLocal() as session:
        for transaction in transactions:
                database_client.insert_transaction(
                    session=session,
                    transaction=transaction,
            )
                
        history = database_client.get_card_history(
            session,
            cc_num=cc_num,
            before=transactions[-1].trans_date_trans_time,
        )

        assert len(history) == 2
        assert list(history.columns) == [
            "cc_num",
            "amt",
            "is_fraud",
            "trans_date_trans_time",
        ]

        assert all(
            history["cc_num"] == cc_num
        )

        assert all(
            history["trans_date_trans_time"]
            < transactions[-1].trans_date_trans_time
        )

        assert history["trans_date_trans_time"].is_monotonic_increasing
        session.rollback()


def test_get_model_version():
    with get_database_access().SessionLocal() as session:
        database_client = DatabaseClient()
        model = database_client.get_model_version(
            session=session,
            model_name="fraud-detection-random-forest",
            model_version="2",
        )

        assert model is not None
        assert model.model_name == "fraud-detection-random-forest"       