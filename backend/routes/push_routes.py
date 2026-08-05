"""Web Push subscription endpoints (VAPID)."""

from flask import Blueprint, jsonify, request

from config import Config
from services import users_service
from services.users_service import ProfileError

push_bp = Blueprint("push", __name__, url_prefix="/api/push")


@push_bp.get("/public-key")
def public_key():
    return jsonify({"public_key": Config.VAPID_PUBLIC_KEY})


@push_bp.post("/subscribe")
def subscribe():
    payload = request.get_json(silent=True) or {}
    user_id = payload.get("user_id")
    subscription = payload.get("subscription")
    if not user_id or not subscription:
        return jsonify({"error": "user_id and subscription are required"}), 400
    try:
        users_service.add_push_subscription(user_id, subscription)
    except ProfileError as exc:
        return jsonify({"error": "validation_failed", "message": str(exc)}), 400
    return jsonify({"subscribed": True})


@push_bp.post("/unsubscribe")
def unsubscribe():
    payload = request.get_json(silent=True) or {}
    user_id = payload.get("user_id")
    endpoint = payload.get("endpoint")
    if not user_id or not endpoint:
        return jsonify({"error": "user_id and endpoint are required"}), 400
    users_service.remove_push_subscription(user_id, endpoint)
    return jsonify({"subscribed": False})
