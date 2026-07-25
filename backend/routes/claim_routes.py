"""Claim submission (member) and review (Discovery employee) endpoints.

Thin handlers — validation, persistence and status transitions live in
`services/claims_service.py`, media handling in `services/storage_service.py`.
"""

import logging

from flask import Blueprint, jsonify, request

from services import storage_service
from services.claims_service import (
    ClaimNotFound,
    ClaimStateError,
    ClaimValidationError,
    approve_claim,
    build_claim_document,
    claim_counts,
    create_claim,
    deny_claim,
    get_claim,
    list_claims,
    validate_submission,
)
from services.members_service import get_employee, get_member, list_employees, list_members

logger = logging.getLogger(__name__)

claim_bp = Blueprint("claims", __name__, url_prefix="/api")

MAX_FILES = 6


def _payload():
    """Accept either multipart (with files) or plain JSON."""
    if request.files or request.form:
        return request.form.to_dict()
    return request.get_json(silent=True) or {}


def _public_claim(claim, include_media_urls=True):
    """Strip Cosmos system fields and sign media for the client."""
    out = {k: v for k, v in claim.items() if not k.startswith("_")}
    if include_media_urls and out.get("media"):
        out["media"] = storage_service.with_read_urls(out["media"])
    return out


@claim_bp.get("/members")
def get_members():
    """Demo member/employee directory. NOT authentication — see members_service."""
    return jsonify({"members": list_members(), "employees": list_employees()})


@claim_bp.get("/suburbs")
def get_suburbs():
    """Known suburb names for the claim form's location field.

    These are the geocoded ones, so a claim filed against any of them lands on
    the hot-spot map immediately. Free text is still accepted on submission —
    a new suburb is geocoded when the claim is approved.
    """
    from services.geocode_service import load_geocache

    query = (request.args.get("q") or "").strip().upper()
    names = sorted(load_geocache().keys())
    if query:
        starts = [n for n in names if n.startswith(query)]
        contains = [n for n in names if query in n and not n.startswith(query)]
        names = starts + contains
    return jsonify({"suburbs": names[:50], "total": len(names)})


@claim_bp.post("/claims")
def submit_claim():
    """Member submits a claim/report. Lands as `status: pending`.

    Accepts multipart/form-data so photos and video ride along with the fields.
    """
    payload = _payload()

    try:
        clean = validate_submission(payload)
    except ClaimValidationError as exc:
        return jsonify({"error": "validation_failed", "fields": exc.args[0]}), 400

    member = get_member(clean["member_id"])
    if not member:
        return jsonify(
            {"error": "validation_failed", "fields": {"member_id": "Unknown member."}}
        ), 400

    files = [f for f in request.files.getlist("media") if f and f.filename]
    if len(files) > MAX_FILES:
        return jsonify(
            {
                "error": "validation_failed",
                "fields": {"media": f"Attach at most {MAX_FILES} files."},
            }
        ), 400

    bad = [f.filename for f in files if not storage_service.is_allowed(f.filename)]
    if bad:
        return jsonify(
            {
                "error": "validation_failed",
                "fields": {"media": f"Unsupported file type: {', '.join(bad)}"},
            }
        ), 400

    # Build the id first so uploaded media is namespaced under the claim it
    # belongs to, then attach whatever landed.
    document = build_claim_document(
        clean,
        member,
        media=[],
        camera_consent=str(payload.get("camera_consent", "")).lower()
        in ("1", "true", "yes", "on"),
    )

    uploaded = []
    if files:
        if not storage_service.is_configured():
            return jsonify(
                {
                    "error": "storage_unavailable",
                    "message": "Media upload isn't configured on the server.",
                }
            ), 503
        try:
            for file_storage in files:
                uploaded.append(
                    storage_service.upload_claim_media(document["id"], file_storage)
                )
        except (ValueError, storage_service.StorageUnavailable) as exc:
            return jsonify({"error": "upload_failed", "message": str(exc)}), 400
    document["media"] = uploaded

    stored = create_claim(document)
    return jsonify({"claim": _public_claim(stored)}), 201


@claim_bp.get("/claims")
def get_claims():
    """List member submissions. Filter with ?status= and/or ?member_id=."""
    status = (request.args.get("status") or "").strip().lower() or None
    member_id = (request.args.get("member_id") or "").strip() or None
    claims = list_claims(status=status, member_id=member_id)
    # Media URLs are signed per claim on the detail view; skip the signing cost
    # (one call per file) when rendering a list.
    return jsonify({"claims": [_public_claim(c, include_media_urls=False) for c in claims]})


@claim_bp.get("/claims/counts")
def get_claim_counts():
    """Queue totals by status, for the employee dashboard."""
    return jsonify(claim_counts())


@claim_bp.get("/claims/<incident_id>")
def get_claim_detail(incident_id):
    try:
        claim = get_claim(incident_id)
    except ClaimNotFound:
        return jsonify({"error": "not_found"}), 404
    return jsonify({"claim": _public_claim(claim)})


def _reviewer():
    """Resolve the acting employee. Mock identity — no auth in the hackathon build."""
    payload = request.get_json(silent=True) or {}
    employee_id = (payload.get("employee_id") or "").strip()
    return get_employee(employee_id), payload


@claim_bp.post("/claims/<incident_id>/approve")
def approve(incident_id):
    """Approve: the claim joins the working dataset and the hot-spot map."""
    employee, payload = _reviewer()
    if not employee:
        return jsonify(
            {"error": "validation_failed", "fields": {"employee_id": "Unknown employee."}}
        ), 400
    try:
        claim = approve_claim(incident_id, employee, note=payload.get("note"))
    except ClaimNotFound:
        return jsonify({"error": "not_found"}), 404
    except ClaimStateError as exc:
        return jsonify({"error": "invalid_state", "message": str(exc)}), 409
    return jsonify({"claim": _public_claim(claim)})


@claim_bp.post("/claims/<incident_id>/deny")
def deny(incident_id):
    """Deny with a reason. The reason is what the member is shown."""
    employee, payload = _reviewer()
    if not employee:
        return jsonify(
            {"error": "validation_failed", "fields": {"employee_id": "Unknown employee."}}
        ), 400
    try:
        claim = deny_claim(incident_id, employee, reason=payload.get("denial_reason"))
    except ClaimValidationError as exc:
        return jsonify({"error": "validation_failed", "fields": exc.args[0]}), 400
    except ClaimNotFound:
        return jsonify({"error": "not_found"}), 404
    except ClaimStateError as exc:
        return jsonify({"error": "invalid_state", "message": str(exc)}), 409
    return jsonify({"claim": _public_claim(claim)})
