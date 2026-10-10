from fraud_detection_processor.payment_client import get_payment


def test_get_payment():
    payment = get_payment()

    assert isinstance(payment, dict)

    assert "columns" in payment
    assert "index" in payment
    assert "data" in payment

    assert isinstance(payment["columns"], list)
    assert isinstance(payment["index"], list)
    assert isinstance(payment["data"], list)

    assert len(payment["data"]) == 1
    assert len(payment["index"]) == 1

    assert len(payment["data"][0]) == len(payment["columns"])