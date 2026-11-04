"""Turn a raw payment into minimised data and an inference payload.

This module has no I/O and no configuration import, so it can be tested
alone. It is the boundary where personal data stops: after
prepare_payment, the card number, names, address, date of birth, gender
and coordinates no longer exist in the processor; only the card
pseudonym (card_id) and the last four digits (pan_last4, for alerts)
are kept.
"""

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from fraud_detection_contracts.inference import (
    CardFeatures,
    FraudDetectionInferenceInput,
    Transaction as InferenceTransaction,
)
from fraud_detection_features.config import CARD_NUM_COLUMN, TIMESTAMP_COLUMN
from fraud_detection_features.privacy import (
    dropped_columns,
    mask_card_number,
    minimize_transactions,
)

# Column of the payment API holding the transaction time (epoch, ms)
PAYMENT_TIME_COLUMN = "current_time"


@dataclass(frozen=True)
class PreparedPayment:
    payment: pd.DataFrame          # one row, minimised (allow-list only)
    transaction_time: datetime
    pan_last4: str
    dropped_columns: list[str]     # names only, for logging


def prepare_payment(payment_data: dict, key: bytes) -> PreparedPayment:
    """Pseudonymise and minimise the payment returned by the payment API."""
    raw = pd.DataFrame(payment_data["data"], columns=payment_data["columns"])

    if len(raw) != 1:
        raise ValueError(f"Expected exactly one payment, got {len(raw)}")

    for column in (CARD_NUM_COLUMN, PAYMENT_TIME_COLUMN):
        if column not in raw.columns:
            raise ValueError(f"Payment has no {column!r} column")

    transaction_time = pd.to_datetime(
        raw.iloc[0][PAYMENT_TIME_COLUMN], unit="ms"
    ).to_pydatetime()

    # The simulator replaces the original timestamp by current_time:
    # it becomes the transaction time used for the card history.
    raw[TIMESTAMP_COLUMN] = transaction_time

    pan_last4 = mask_card_number(raw.iloc[0][CARD_NUM_COLUMN])
    removed = dropped_columns(raw)

    payment_minimized = minimize_transactions(raw, key)

    return PreparedPayment(
        payment=payment_minimized,
        transaction_time=transaction_time,
        pan_last4=pan_last4,
        dropped_columns=removed,
    )


def numpy_to_python(value: object) -> object:
    """Convert a numpy scalar (numpy.int64, numpy.float64...) to Python."""
    return value.item() if hasattr(value, "item") else value


def build_inference_input(features_row: pd.Series) -> FraudDetectionInferenceInput:
    """Build the API payload from the features row, field by field.

    Only the fields declared by the contract are taken: a column added
    upstream is never sent by accident.
    Pydantic v2 accepts some numpy types as fields but...Sometime not
    --> so numpy_to_python
    """
    def fields_of(model) -> dict:
        return {name: numpy_to_python(features_row[name]) for name in model.model_fields}

    return FraudDetectionInferenceInput(
        transaction=InferenceTransaction(**fields_of(InferenceTransaction)),
        card_features=CardFeatures(**fields_of(CardFeatures)),
    )
