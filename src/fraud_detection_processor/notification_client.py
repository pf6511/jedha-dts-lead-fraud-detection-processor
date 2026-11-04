import logging

import httpx

from fraud_detection_processor.config import DISCORD_FRAUD_WEBHOOK_URL

logger = logging.getLogger(__name__)


def build_fraud_alert_message(
    *,
    transaction_id: str,
    pan_last4: str,
    amount: float,
    fraud_probability: float,
    model_version: str,
) -> str:
    """Alert content: masked card only (last four digits), no personal data."""
    return "\n".join(
        [
            "** potential fraud detected **",
            f"**Transaction ID :** `{transaction_id}`",
            f"**Card :** `**** {pan_last4}`",
            f"**Amount :** `{amount:.2f}`",
            f"**Fraud probability :** `{fraud_probability:.2%}`",
            f"**Model version :** `{model_version}`",
        ]
    )


def notify_fraud_detected(
    *,
    transaction_id: str,
    pan_last4: str,
    amount: float,
    fraud_probability: float,
    model_version: str,
) -> bool:
    """Send a fraud alert to the configured Discord channel."""
    payload = {
        "content": build_fraud_alert_message(
            transaction_id=transaction_id,
            pan_last4=pan_last4,
            amount=amount,
            fraud_probability=fraud_probability,
            model_version=model_version,
        ),
        "allowed_mentions": {"parse": []},
    }

    try:
        response = httpx.post(DISCORD_FRAUD_WEBHOOK_URL, json=payload, timeout=10.0)
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
