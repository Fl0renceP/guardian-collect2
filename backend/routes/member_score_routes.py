"""Member-facing Guardian Safety Score.

Kept in its own blueprint rather than added to `safety_routes.py`, which is
Victoria's claim-risk scorer — same phrase, different subject, and separating
them keeps merges clean.
"""

import logging

from flask import Blueprint, jsonify

from services.member_score_service import calculate_member_score

logger = logging.getLogger(__name__)

member_score_bp = Blueprint("member_score", __name__, url_prefix="/api")


@member_score_bp.get("/members/<member_id>/safety-score")
def get_member_safety_score(member_id):
    """The member's reward score, with the full breakdown behind it."""
    result = calculate_member_score(member_id)
    if result is None:
        return jsonify({"error": "not_found", "message": "Unknown member."}), 404
    return jsonify(result)
