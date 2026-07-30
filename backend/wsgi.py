"""WSGI entrypoint for Gunicorn.

Prefer package-qualified imports in cloud runtimes where the working directory
is the repository root, but keep a local fallback for direct backend execution.
"""

try:
	from backend.app import app
except ModuleNotFoundError:
	from app import app
