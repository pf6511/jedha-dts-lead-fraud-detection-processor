import pytest

from fraud_detection_processor.validation import quarantine_payload, validate_payment

from payment_samples import CARD_NUMBER, PERSONAL_COLUMNS, payment_api_response


def test_valid_payment_has_no_error():
    assert validate_payment(payment_api_response()) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("amt", 0),
        ("amt", -12.5),
        ("cc_num", 1234),
        ("state", "Illinois"),
        ("city_pop", -1),
        ("is_fraud", 2),
        ("lat", 123.0),
        ("merchant", ""),
        ("lat", None),
    ],
)
def test_invalid_value_is_reported(field, value):
    errors = validate_payment(payment_api_response(**{field: value}))
    assert any(error.startswith(f"{field}:") for error in errors)


def test_missing_column_is_reported():
    errors = validate_payment(payment_api_response(drop=("amt",)))
    assert any(error.startswith("amt:") for error in errors)


def test_several_payments_are_reported():
    errors = validate_payment(payment_api_response(rows=2))
    assert errors and errors[0].startswith("payment:")


def test_errors_do_not_contain_the_rejected_value():
    wrong_card = 40001234123412345678999        # too long
    errors = validate_payment(payment_api_response(cc_num=wrong_card, amt=-99.99))

    text = " ".join(errors)
    assert str(wrong_card) not in text
    assert "-99.99" not in text


def test_quarantine_payload_has_no_personal_data():
    payload = quarantine_payload(payment_api_response(amt=-5))

    for column in PERSONAL_COLUMNS:
        assert column not in payload
    assert str(CARD_NUMBER) not in str(payload)
    assert payload["amt"] == -5
    assert payload["trans_num"] == "abc123"

def test_eleven_digit_card_number_is_valid():
    assert validate_payment(payment_api_response(cc_num=60416207185)) == []
