"""Central alert policy and in-memory delivery for demo notifications."""

from collections import deque
from datetime import datetime, timezone
from queue import Empty, Queue
import threading
import uuid


_ALLOWED_STATUS = {"verified", "suspect", "offender"}
_LEVEL_ORDER = {"low": 1, "medium": 2, "high": 3}


def _utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _status_level(status):
    if status == "offender":
        return "high"
    if status == "suspect":
        return "medium"
    return "low"


def _status_severity(status):
    if status == "offender":
        return "critical"
    if status == "suspect":
        return "warning"
    return "info"


def _eligible_by_min_level(status, min_level):
    required = _LEVEL_ORDER.get(min_level, _LEVEL_ORDER["medium"])
    current = _LEVEL_ORDER[_status_level(status)]
    return current >= required


class AlertDeliveryService:
    def __init__(self, max_events=250):
        self._events = deque(maxlen=max_events)
        self._subscribers = []
        self._lock = threading.Lock()

    @staticmethod
    def _delivery_policy(status):
        return {
            "member": {
                "alert": status == "offender",
                "push": status == "offender",
            },
            "crime_prevention": {
                "alert": status in {"suspect", "offender"},
                "push": False,
            },
        }

    def register_detection(
        self,
        *,
        status,
        entity_type,
        entity,
        source_endpoint,
        detection_mode="manual_scan",
        push_enabled=False,
        push_dry_run=True,
        push_min_level="medium",
    ):
        normalized_status = (status or "").strip().lower()
        if normalized_status not in _ALLOWED_STATUS:
            return None

        delivery = self._delivery_policy(normalized_status)

        # Keep legacy min-level tuning as a global push cutoff on top of policy.
        if not _eligible_by_min_level(normalized_status, push_min_level):
            delivery["member"]["push"] = False

        event = {
            "id": str(uuid.uuid4()),
            "timestamp_utc": _utc_now_iso(),
            "category": normalized_status,
            "severity": _status_severity(normalized_status),
            "level": _status_level(normalized_status),
            "entity_type": entity_type,
            "entity": entity,
            "detection_mode": detection_mode,
            "source_endpoint": source_endpoint,
            "title": f"{normalized_status.upper()} detected",
            "message": f"{entity_type.capitalize()} matched as {normalized_status}.",
            "delivery": delivery,
            "push": {
                "enabled": bool(push_enabled),
                "dry_run": bool(push_dry_run),
                "attempted": bool(push_enabled and delivery["member"]["push"]),
                "delivered": bool(push_enabled and not push_dry_run and delivery["member"]["push"]),
            },
        }

        with self._lock:
            self._events.appendleft(event)
            subscribers = list(self._subscribers)

        for sub in subscribers:
            if self._matches_subscription(event, sub["audience"], sub["channel"]):
                try:
                    sub["queue"].put_nowait(event)
                except Exception:
                    # Best-effort live feed: drop if subscriber is saturated/disconnected.
                    pass

        return event

    @staticmethod
    def _matches_subscription(event, audience, channel):
        audience = (audience or "crime_prevention").strip().lower()
        channel = (channel or "alerts").strip().lower()
        if audience not in {"member", "crime_prevention"}:
            return False
        if channel not in {"alerts", "push"}:
            return False
        return bool(event.get("delivery", {}).get(audience, {}).get("alert" if channel == "alerts" else "push"))

    def list_events(self, *, audience="crime_prevention", channel="alerts", limit=50):
        audience = (audience or "crime_prevention").strip().lower()
        channel = (channel or "alerts").strip().lower()
        if audience not in {"member", "crime_prevention"}:
            return []
        if channel not in {"alerts", "push"}:
            return []

        key = "alert" if channel == "alerts" else "push"
        cap = max(1, min(int(limit), 200))

        with self._lock:
            items = [e for e in self._events if e.get("delivery", {}).get(audience, {}).get(key)]

        return items[:cap]

    def subscribe(self, *, audience="crime_prevention", channel="alerts"):
        q = Queue(maxsize=100)
        record = {
            "audience": (audience or "crime_prevention").strip().lower(),
            "channel": (channel or "alerts").strip().lower(),
            "queue": q,
        }
        with self._lock:
            self._subscribers.append(record)
        return record

    def unsubscribe(self, subscription):
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s is not subscription]

    @staticmethod
    def queue_get(subscription, timeout_seconds=20):
        return subscription["queue"].get(timeout=timeout_seconds)


alert_delivery_service = AlertDeliveryService()
