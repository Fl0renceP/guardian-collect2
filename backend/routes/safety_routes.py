from flask import Blueprint, jsonify
from services.claims_service import load_claims
from services.safety_score import calculate_safety_score


safety_bp = Blueprint("safety", __name__)


@safety_bp.route("/api/safety-score/<claim_id>")
def safety_score(claim_id):

    claims = load_claims()

    claim = next(
        (c for c in claims if c["incident"] == claim_id),
        None
    )

    if claim is None:
        return jsonify({
            "error": "Claim not found"
        }), 404


    result = calculate_safety_score(claim)

    return jsonify({
        "claim_id": claim_id,
        **result
    })