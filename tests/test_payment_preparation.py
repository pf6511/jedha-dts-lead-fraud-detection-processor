"""Tests of the personal data boundary of the processor (no I/O, no secrets.env)."""

import json
import re

import pandas as pd
import pytest

from fraud_detection_features.features import build_inference_features
from fraud_detection_features.privacy import KEPT_COLUMNS, pseudonymize_card_number

from fraud_detection_processor.payment_preparation import (
    build_inference_input,
    payload_to_payment,
    payment_to_payload,
    prepare_payment,
)

from payment_samples import CARD_NUMBER, KEY, PERSONAL_COLUMNS, payment_api_response


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
    with pytest.raises(ValueError, match="cc_num"):
        prepare_payment(payment_api_response(drop=("cc_num",)), KEY)


def test_inference_payload_follows_the_contract_without_personal_data():
    prepared = prepare_payment(payment_api_response(), KEY)
    card_id = prepared.payment.iloc[0]["card_id"]
    history = pd.DataFrame(
        {
            "card_id": [card_id] * 2,
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
    assert str(CARD_NUMBER) not in sent
    assert card_id not in sent
    for column in PERSONAL_COLUMNS:
        assert f'"{column}"' not in sent


def test_failed_payment_can_be_stored_and_rebuilt():
    prepared = prepare_payment(payment_api_response(), KEY)

    payload = payment_to_payload(prepared)
    stored = json.dumps(payload)             # what goes to the JSONB column
    assert str(CARD_NUMBER) not in stored

    rebuilt = payload_to_payment(json.loads(stored))

    assert rebuilt.transaction_time == prepared.transaction_time
    assert rebuilt.pan_last4 == "1234"
    assert rebuilt.payment.iloc[0]["card_id"] == prepared.payment.iloc[0]["card_id"]
    assert set(rebuilt.payment.columns) == set(prepared.payment.columns)
