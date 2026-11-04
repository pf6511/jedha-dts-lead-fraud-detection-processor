import pytest

from fraud_detection_features.privacy import KEPT_COLUMNS

from fraud_detection_processor.config import CARD_ID_KEY
from fraud_detection_processor.payment_client import get_payment
from fraud_detection_processor.payment_preparation import prepare_payment

pytestmark = pytest.mark.integration


def test_get_payment_and_minimise_it():
    """One real call (the payment API is limited to 5 calls per minute)."""
    payment = get_payment()

    # Shape of the payment API response (pandas "split" orientation)
    assert isinstance(payment, dict)
    for key in ("columns", "index", "data"):
        assert isinstance(payment[key], list)
    assert len(payment["data"]) == 1
    assert len(payment["index"]) == 1
    assert len(payment["data"][0]) == len(payment["columns"])

    # The real payload contains personal data...
    assert "cc_num" in payment["columns"]

    # ...which does not survive minimisation
    prepared = prepare_payment(payment, CARD_ID_KEY)
    assert set(prepared.payment.columns) <= set(KEPT_COLUMNS)
    assert "cc_num" in prepared.dropped_columns
    assert len(prepared.pan_last4) == 4
