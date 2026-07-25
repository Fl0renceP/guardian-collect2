"""Liveness and data-readiness checks."""

from flask import Blueprint, jsonify

from services.claims_service import source_status
from services.geocode_service import geocache_status

health_bp = Blueprint("health", __name__, url_prefix="/api")


@health_bp.get("/health")
def health():
    """Reports the live claims source and geocode coverage too — the map looks
    'broken' when it's really just incomplete or serving a stale fallback."""
    return jsonify(
        {
            "status": "ok",
            "service": "Guardian Collective API",
            "claims": source_status(),
            "geocode": geocache_status(),
        }
    )
