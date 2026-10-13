import os
from uuid import uuid4

import pytest


def test_notification_client_sends_real_discord_message() -> None:
    if os.getenv("RUN_DISCORD_INTEGRATION_TESTS") != "1":
        pytest.skip(
            "Set RUN_DISCORD_INTEGRATION_TESTS=1 to send a real Discord alert."
        )

    from fraud_detection_processor.notification_client import (
        notify_fraud_detected,
    )

    success = notify_fraud_detected(
        transaction_id=uuid4().int,
        amount=123.45,
        fraud_probability=0.99,
    )

    assert success is True, (
        "Discord notification failed. Check the processor logs "
        "and the local webhook configuration."
    )