"""User directory and member profile endpoints.

No authentication yet — these read and write whatever identity the caller names.
The `auth` block on each user document is stripped before it leaves the server.
"""

import logging

from flask import Blueprint, jsonify, request

from services import users_service
from services.users_service import ProfileError

logger = logging.getLogger(__name__)

user_bp = Blueprint("users", __name__, url_prefix="/api")


@user_bp.get("/users")
def get_users():
    """The directory. Filter with ?role=member|employee|cpu."""
    role = (request.args.get("role") or "").strip().lower() or None
    if role and role not in users_service.ROLES:
        return jsonify({"error": f"role must be one of {', '.join(users_service.ROLES)}"}), 400
    return jsonify(
        {
            "users": users_service.list_users(role=role),
            "counts": users_service.counts(),
        }
    )


@user_bp.get("/users/<user_id>")
def get_one_user(user_id):
    user = users_service.get_user(user_id)
    if not user:
        return jsonify({"error": "not_found"}), 404
    return jsonify({"user": user})


@user_bp.patch("/users/<user_id>/location")
def update_location(user_id):
    """Set, update or clear a member's home location.

    Body: {address?, suburb?, lat?, lng?, share_location?, alert_radius_km?}

    Sending `share_location: false` clears the stored coordinates outright —
    withdrawing permission removes the data rather than just ignoring it.
    """
    payload = request.get_json(silent=True) or {}
    try:
        user = users_service.update_member_location(
            user_id,
            address=payload.get("address"),
            suburb=payload.get("suburb"),
            lat=payload.get("lat"),
            lng=payload.get("lng"),
            share_location=payload.get("share_location"),
            alert_radius_km=payload.get("alert_radius_km"),
        )
    except ProfileError as exc:
        return jsonify({"error": "validation_failed", "message": str(exc)}), 400
    return jsonify({"user": user})
