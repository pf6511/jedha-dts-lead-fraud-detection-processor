"""Validation of the raw payment, before any processing.

A payment that fails the validation is quarantined: it is stored in
quarantined_payments, without personal data, and is not processed.

The error messages keep the field name and the rule, never the
rejected value (a wrong card number must not end up in a table).
"""

from pydantic import BaseModel, Field, ValidationError

# Non-personal fields kept for a quarantined payment
QUARANTINE_FIELDS = [
    "trans_num", "merchant", "category", "amt", "city_pop",
    "state", "job", "is_fraud", "current_time",
]


class RawPayment(BaseModel):
    """Fields of the raw payment used by the pipeline, with their rules."""

    trans_num: str = Field(min_length=1)
    cc_num: int = Field(ge=10**7, lt=10**19)       # 8 to 19 digits
    current_time: int = Field(gt=0)                  # epoch, milliseconds

    amt: float = Field(gt=0)
    merchant: str = Field(min_length=1)
    category: str = Field(min_length=1)
    city_pop: int = Field(ge=0)
    state: str = Field(pattern=r"^[A-Z]{2}$")
    job: str = Field(min_length=1)

    # Used to compute the distance, then removed by the minimisation
    lat: float = Field(ge=-90, le=90)
    long: float = Field(ge=-180, le=180)
    merch_lat: float = Field(ge=-90, le=90)
    merch_long: float = Field(ge=-180, le=180)

    is_fraud: int = Field(ge=0, le=1)


def payment_row(payment_data: dict) -> dict:
    """The payment as a dict {column: value}; the API returns one row."""
    if len(payment_data["data"]) != 1:
        raise ValueError(f"expected one payment, got {len(payment_data['data'])}")
    return dict(zip(payment_data["columns"], payment_data["data"][0]))


def validate_payment(payment_data: dict) -> list[str]:
    """Return the validation errors; an empty list means the payment is valid."""
    try:
        row = payment_row(payment_data)
    except (KeyError, TypeError, ValueError) as exc:
        return [f"payment: malformed response ({exc})"]

    try:
        RawPayment.model_validate(row)
    except ValidationError as exc:
        # include_input=False: the rejected values are not kept
        return [
            f"{error['loc'][0]}: {error['msg']}"
            for error in exc.errors(include_input=False)
        ]

    return []


def quarantine_payload(payment_data: dict) -> dict:
    """Non-personal fields of the payment, for the quarantine table."""
    try:
        row = payment_row(payment_data)
    except (KeyError, TypeError, ValueError):
        return {}
    return {field: row[field] for field in QUARANTINE_FIELDS if field in row}
