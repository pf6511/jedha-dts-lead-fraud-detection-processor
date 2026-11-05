"""Test data: a fictitious payment, shaped like the payment API answer."""

KEY = b"k" * 32
CARD_NUMBER = 4000123412341234      # fictitious card number

PERSONAL_COLUMNS = [
    "cc_num", "first", "last", "street", "city", "zip", "dob",
    "gender", "lat", "long", "merch_lat", "merch_long",
]

PAYMENT = {
    "cc_num": CARD_NUMBER,
    "merchant": "fraud_Kilback LLC",
    "category": "grocery_pos",
    "amt": 120.5,
    "first": "Jane",
    "last": "Doe",
    "gender": "F",
    "street": "1 Main Street",
    "city": "Springfield",
    "state": "IL",
    "zip": 62701,
    "lat": 39.78,
    "long": -89.65,
    "city_pop": 116250,
    "job": "Teacher",
    "dob": "1980-01-01",
    "trans_num": "abc123",
    "merch_lat": 39.90,
    "merch_long": -89.50,
    "is_fraud": 0,
    "current_time": 1730736000000,   # 2024-11-04 16:00 UTC
}


def payment_api_response(rows: int = 1, drop: tuple = (), **changes) -> dict:
    """Payment API answer (pandas "split" orientation).

    changes: values to replace, e.g. amt=-5; drop: columns to remove.
    """
    payment = {**PAYMENT, **changes}
    columns = [column for column in payment if column not in drop]
    row = [payment[column] for column in columns]
    return {"columns": columns, "index": list(range(rows)), "data": [row] * rows}
