"""Lazy Cosmos DB container handle.

Built on first use rather than at import, so the app still boots (and the CSV
fallback still works) when credentials are missing or the account is down.
"""

import logging
import threading

from config import Config

logger = logging.getLogger(__name__)

_container = None
_lock = threading.Lock()


class CosmosUnavailable(RuntimeError):
    """Cosmos isn't configured or can't be reached."""


def is_configured():
    return bool(Config.COSMOS_URI and Config.COSMOS_KEY)


def get_container():
    """Return the claims container client, building it once. Thread-safe.

    Raises CosmosUnavailable rather than letting the azure SDK's exception types
    leak into callers, so claims_service can decide whether to fall back.
    """
    global _container
    if _container is not None:
        return _container

    if not is_configured():
        raise CosmosUnavailable("COSMOS_URI / COSMOS_KEY are not set")

    with _lock:
        if _container is not None:
            return _container
        try:
            from azure.cosmos import CosmosClient
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise CosmosUnavailable(f"azure-cosmos is not installed: {exc}") from exc

        try:
            client = CosmosClient(Config.COSMOS_URI, credential=Config.COSMOS_KEY)
            _container = (
                client.get_database_client(Config.COSMOS_DATABASE)
                .get_container_client(Config.COSMOS_CONTAINER)
            )
        except Exception as exc:
            raise CosmosUnavailable(f"could not connect to Cosmos: {exc}") from exc

        logger.info(
            "Cosmos container ready: %s/%s", Config.COSMOS_DATABASE, Config.COSMOS_CONTAINER
        )
        return _container
