"""Deployment entrypoint shim.

Azure/Oryx often defaults to `gunicorn app:app` from repository root. This file
re-exports the Flask app from backend/app.py so that startup command remains
resolvable even when module detection is imperfect.
"""

from backend.app import app
