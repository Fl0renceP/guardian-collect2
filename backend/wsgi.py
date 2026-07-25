"""Entrypoint for gunicorn / Azure App Service: `gunicorn wsgi:app`."""

from app import create_app

app = create_app()
