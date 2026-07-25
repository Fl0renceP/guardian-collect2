"""Guardian Collective backend — Flask app factory."""

import logging

from flask import Flask, send_from_directory

from config import Config
from routes.claim_routes import claim_bp
from routes.health_routes import health_bp
from routes.hotspot_routes import hotspot_bp
from routes.route_routes import route_bp
from services.claims_service import warm_cache


def create_app(config_object=Config):
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.config.from_object(config_object)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # The Azure SDKs log every request/response header at INFO — useful when
    # debugging them, unreadable otherwise.
    for noisy in ("azure", "azure.core.pipeline.policies.http_logging_policy", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    app.register_blueprint(health_bp)
    app.register_blueprint(hotspot_bp)
    app.register_blueprint(claim_bp)
    app.register_blueprint(route_bp)

    # Pull the claims collection on boot (off-thread) so the first visitor
    # doesn't pay the cold-load cost.
    warm_cache()

    @app.get("/")
    def index():
        """The hot-spot map — the first thing a user sees on entering the app."""
        return send_from_directory(app.static_folder, "index.html")

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, port=5000)
