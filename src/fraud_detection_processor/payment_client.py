import httpx
import json

from fraud_detection_processor.config import PAYMENT_API_URL


def get_payment() -> dict:
    response = httpx.get(
        PAYMENT_API_URL,
        timeout=10.0,
    )
    response.raise_for_status()
    return json.loads(response.json())