import logging

import httpx

from fraud_detection_processor.config import (
    DISCORD_FRAUD_WEBHOOK_URL,
)

logger = logging.getLogger(__name__)


def notify_fraud_detected(
    *,
    transaction_id: int,
    amount: float,
    fraud_probability: float,
) -> bool:
    """Send a fraud alert to the configured Discord channel."""

    message = "\n".join(
        [
            "** potential fraud detected **",
            f"**Transaction ID :** `{transaction_id}`",
            f"**Amount :** `{amount:.2f}`",
            f"**fraud probability :** `{fraud_probability:.2%}`",
        ]
    )

    payload = {
        "content": message,
        "allowed_mentions": {"parse": []},
    }

    try:
        response = httpx.post(
            DISCORD_FRAUD_WEBHOOK_URL,
            json=payload,
            timeout=10.0,
        )
        response.raise_for_status()
        return True

    except httpx.HTTPError as exc:
        # A notification failure must not stop payment processing.
        logger.error(
            "Discord fraud alert failed for transaction_id=%s (%s)",
            transaction_id,
            type(exc).__name__,
        )
        return False