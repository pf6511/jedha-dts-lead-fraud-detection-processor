import pandas as pd
import pytest
from pathlib import Path
    

def create_test_dataset() -> None:

        DATA_PATH = Path("../data/raw/fraudTest.csv")
        TEST_DATA_PATH = Path("tests/data/transactions_test.csv")

        df = pd.read_csv(DATA_PATH,index_col=0)
        print("len(df) : " , len(df))
        selected_cards = (
            df.groupby("cc_num")
            .size()
            .loc[lambda s: s >= 4]
            .head(2)
            .index
        )

        test_df = (
            df[df["cc_num"].isin(selected_cards)]
            .sort_values(["cc_num", "trans_date_trans_time"])
        )

        test_df.to_csv(TEST_DATA_PATH, index=False)


def create_test_dataset_with_fraud() -> None:

        DATA_PATH = Path("../data/raw/fraudTest.csv")
        TEST_DATA_PATH = Path("tests/data/transactions_test_with_fraud.csv")

        df = pd.read_csv(DATA_PATH,index_col=0)
        print("len(df) : " , len(df))

        df["trans_date_trans_time"] = pd.to_datetime(
                df["trans_date_trans_time"]
        )
        card_stats = (
            df.groupby("cc_num")
            .agg(
                transaction_count=("cc_num", "size"),
                fraud_count=("is_fraud", "sum"),
            )
        )

        card_stats["fraud_rate"] = (
            card_stats["fraud_count"]
            / card_stats["transaction_count"]
        )

        selected_cards = (
            card_stats[
                (card_stats["transaction_count"] >= 2)
                & (card_stats["fraud_rate"] > 0)
                & (card_stats["fraud_rate"] < 1.0)
            ]
            .sort_values("fraud_rate", ascending=False)
            .head(4)
            .index
        )

        test_df = (
            df[df["cc_num"].isin(selected_cards)]
                .sort_values(["cc_num", "trans_date_trans_time"])
        )

        print("Selected cards:")
        print(card_stats.loc[selected_cards])

        test_df.to_csv(TEST_DATA_PATH, index=False)

if __name__ == "__main__":
    create_test_dataset_with_fraud()