import os
from uuid import uuid4

import pytest

from fraud_detection_processor.notification_client import (
    build_fraud_alert_message,
    notify_fraud_detected,
)

pytestmark = pytest.mark.integration


def test_alert_message_shows_masked_card_only():
    message = build_fraud_alert_message(
        transaction_id="0b6f5c1e-0000-0000-0000-000000000000",
        pan_last4="1234",
        amount=123.45,
        fraud_probability=0.99,
        model_version="9",
    )

    assert "**** 1234" in message
    assert "123.45" in message
    assert "99.00%" in message


def test_notification_client_sends_real_discord_message() -> None:
    if os.getenv("RUN_DISCORD_INTEGRATION_TESTS") != "1":
        pytest.skip("Set RUN_DISCORD_INTEGRATION_TESTS=1 to send a real Discord alert.")

    success = notify_fraud_detected(
        transaction_id=str(uuid4()),
        pan_last4="0000",
        amount=123.45,
        fraud_probability=0.99,
        model_version="test",
    )

    assert success is True, (
        "Discord notification failed. Check the processor logs "
        "and the local webhook configuration."
    )
