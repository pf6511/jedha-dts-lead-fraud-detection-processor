"""Tests of the personal data boundary of the processor (no I/O, no secrets.env)."""

import json
import re

import pandas as pd
import pytest

from fraud_detection_features.features import build_inference_features
from fraud_detection_features.privacy import KEPT_COLUMNS, pseudonymize_card_number

from fraud_detection_processor.payment_preparation import (
    build_inference_input,
    prepare_payment,
)

KEY = b"k" * 32
CARD_NUMBER = "4000123412341234"   # fake card number

PERSONAL_COLUMNS = [
    "cc_num", "first", "last", "street", "city", "zip", "dob",
    "gender", "lat", "long", "merch_lat", "merch_long",
]


def payment_api_response(rows: int = 1) -> dict:
    """Same shape as the payment API: pandas 'split' orientation."""
    columns = [
        "cc_num", "merchant", "category", "amt", "first", "last", "gender",
        "street", "city", "state", "zip", "lat", "long", "city_pop", "job",
        "dob", "trans_num", "merch_lat", "merch_long", "is_fraud", "current_time",
    ]
    row = [
        CARD_NUMBER, "fraud_Kilback LLC", "grocery_pos", 120.5, "Jane", "Doe", "F",
        "1 Main Street", "Springfield", "IL", 62701, 39.78, -89.65, 116250,
        "Teacher", "1980-01-01", "abc123", 39.90, -89.50, 0, 1730736000000,
    ]
    return {"columns": columns, "data": [row] * rows}


def test_minimised_payment_contains_no_personal_data():
    prepared = prepare_payment(payment_api_response(), KEY)

    assert set(prepared.payment.columns) <= set(KEPT_COLUMNS)
    for column in PERSONAL_COLUMNS:
        assert column not in prepared.payment.columns

    # current_time is replaced by the transaction timestamp column
    assert "current_time" not in prepared.payment.columns
    assert set(prepared.dropped_columns) >= set(PERSONAL_COLUMNS)


def test_card_is_pseudonymised_and_masked():
    prepared = prepare_payment(payment_api_response(), KEY)
    card_id = prepared.payment.iloc[0]["card_id"]

    assert re.fullmatch(r"[0-9a-f]{64}", card_id)
    assert card_id == pseudonymize_card_number(CARD_NUMBER, KEY)
    assert prepared.pan_last4 == "1234"


def test_transaction_time_comes_from_current_time():
    prepared = prepare_payment(payment_api_response(), KEY)

    assert prepared.transaction_time == pd.Timestamp("2024-11-04 16:00:00").to_pydatetime()
    assert prepared.payment.iloc[0]["trans_date_trans_time"] == prepared.transaction_time


def test_more_than_one_payment_is_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        prepare_payment(payment_api_response(rows=2), KEY)


def test_payment_without_card_number_is_rejected():
    response = payment_api_response()
    index = response["columns"].index("cc_num")
    response["columns"].pop(index)
    response["data"] = [row[:index] + row[index + 1:] for row in response["data"]]

    with pytest.raises(ValueError, match="cc_num"):
        prepare_payment(response, KEY)


def test_inference_payload_follows_the_contract_without_personal_data():
    prepared = prepare_payment(payment_api_response(), KEY)
    history = pd.DataFrame(
        {
            "card_id": [prepared.payment.iloc[0]["card_id"]] * 2,
            "amt": [10.0, 20.0],
            "is_fraud": [0, 1],
            "trans_date_trans_time": pd.to_datetime(["2024-11-01", "2024-11-02"]),
        }
    )

    features = build_inference_features(prepared.payment, history, p0=0.004, alpha=10.0)
    payload = build_inference_input(features.iloc[0])

    assert payload.card_features.card_transaction_count == 2
    assert payload.card_features.card_fraud_count == 1

    sent = json.dumps(payload.model_dump())
    assert CARD_NUMBER not in sent
    assert prepared.payment.iloc[0]["card_id"] not in sent
    for column in PERSONAL_COLUMNS:
        assert f'"{column}"' not in sent
