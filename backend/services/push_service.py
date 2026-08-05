"""Web Push delivery for detection alerts (VAPID, no third-party vendor).

Fire-and-forget by design: a failed push must never break a scan response.
Expired subscriptions (404/410 from the browser's push service) are removed so
dead endpoints stop being retried.
"""

import json
import logging

from config import Config
from services import users_service

logger = logging.getLogger(__name__)


def _send_one(user_id, subscription, payload):
    try:
        from pywebpush import webpush, WebPushException
    except ImportError:
        logger.warning("pywebpush not installed \u2014 skipping push delivery.")
        return

    try:
        webpush(
            subscription_info=subscription,
            data=json.dumps(payload),
            vapid_private_key=Config.VAPID_PRIVATE_KEY,
            vapid_claims={"sub": Config.VAPID_CLAIMS_EMAIL},
        )
    except WebPushException as exc:
        status = getattr(exc.response, "status_code", None)
        if status in (404, 410):
            users_service.remove_push_subscription(user_id, subscription.get("endpoint"))
        else:
            logger.warning("Push delivery failed for %s: %s", user_id, exc)


def notify_detection(event):
    """Push `event` (from alerts_service.record_detection) to every subscriber
    in its audience \u2014 the same roles already allowed to see it in-app."""
    if not event or not Config.VAPID_PRIVATE_KEY:
        return

    payload = {
        "title": event["title"],
        "body": event["detail"],
        "url": "/alerts",
        "severity": event["severity"],
    }
    for role in event.get("push_audience") or event.get("audience") or []:
        for user_id, subscription in users_service.list_push_subscriptions(role):
            _send_one(user_id, subscription, payload)
