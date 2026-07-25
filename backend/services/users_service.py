"""User directory backed by Cosmos DB.

One `users` container holds all three stakeholder types, discriminated by
`role`: `member`, `employee`, `cpu`. Role-specific fields live in a nested
profile (`member_profile` / `employee_profile` / `unit_profile`) so the shared
identity fields — email, name, phone, status — stay in one place and
authentication, when it lands, has a single object to attach to.

**Partition key is `/role`.** The dominant reads are "list everyone of this
role" (the role switcher, the review queue's assessor list, the unit picker),
and those become single-partition queries. The trade-off is only three logical
partitions, which would hot-spot at serious write throughput — at that scale
you'd repartition on `/id` and add an email→id lookup. For this project the
container is tiny and the query pattern wins.

Auth is NOT implemented here. `auth` is a placeholder block on each document so
the shape is ready; nothing verifies anything yet. See README → Known gaps.
"""

import logging
import threading
import time
from datetime import datetime, timezone

from config import Config
from services.cosmos_client import CosmosUnavailable, is_configured

logger = logging.getLogger(__name__)

ROLES = ("member", "employee", "cpu")

_cache = None
_cache_at = 0.0
_container = None
# Two locks, deliberately. The cache refresh calls _users_container() while
# holding its lock, so sharing one non-reentrant lock between them deadlocks.
_cache_lock = threading.Lock()
_container_lock = threading.Lock()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _users_container():
    """The users container, created on first use. Built once per process."""
    global _container
    if _container is not None:
        return _container

    if not is_configured():
        raise CosmosUnavailable("COSMOS_URI / COSMOS_KEY are not set")

    with _container_lock:
        if _container is not None:
            return _container
        try:
            from azure.cosmos import CosmosClient, PartitionKey

            client = CosmosClient(Config.COSMOS_URI, credential=Config.COSMOS_KEY)
            database = client.get_database_client(Config.COSMOS_DATABASE)
            _container = database.create_container_if_not_exists(
                id=Config.COSMOS_USERS_CONTAINER,
                partition_key=PartitionKey(path="/role"),
            )
        except Exception as exc:
            raise CosmosUnavailable(f"could not open the users container: {exc}") from exc
        logger.info("Users container ready: %s", Config.COSMOS_USERS_CONTAINER)
        return _container


def _load(force=False):
    """All users, cached briefly. The directory is small and read constantly."""
    global _cache, _cache_at

    fresh = _cache is not None and (time.monotonic() - _cache_at) < Config.USERS_CACHE_TTL_SECONDS
    if fresh and not force:
        return _cache

    # Resolve the container before taking the cache lock — it takes its own.
    container = _users_container()

    with _cache_lock:
        if not force and _cache is not None and (
            time.monotonic() - _cache_at
        ) < Config.USERS_CACHE_TTL_SECONDS:
            return _cache

        try:
            users = list(
                container.query_items(
                    "SELECT * FROM c ORDER BY c.user_id", enable_cross_partition_query=True
                )
            )
        except Exception as exc:
            if _cache is not None:
                logger.exception("Users refresh failed — serving the previous snapshot.")
                return _cache
            raise CosmosUnavailable(f"could not read users: {exc}") from exc

        _cache = users
        _cache_at = time.monotonic()
        logger.info("Loaded %d users from Cosmos", len(users))
        return _cache


def invalidate():
    global _cache_at
    _cache_at = 0.0


def _public(user):
    """Strip Cosmos system fields and anything auth-related before returning."""
    return {k: v for k, v in user.items() if not k.startswith("_") and k != "auth"}


def list_users(role=None):
    users = _load()
    if role:
        users = [u for u in users if u.get("role") == role]
    return [_public(u) for u in users]


def get_user(user_id):
    for user in _load():
        if user.get("user_id") == user_id:
            return _public(user)
    return None


def get_user_raw(user_id):
    for user in _load():
        if user.get("user_id") == user_id:
            return user
    return None


def upsert_user(document):
    """Write a user. Used by the seed script and profile updates."""
    container = _users_container()
    document.setdefault("id", document["user_id"])
    document.setdefault("created_at", _now())
    document["updated_at"] = _now()
    try:
        stored = container.upsert_item(document)
    except Exception as exc:
        raise CosmosUnavailable(f"could not save user: {exc}") from exc
    invalidate()
    return stored


def counts():
    users = _load()
    out = {role: 0 for role in ROLES}
    for user in users:
        out[user.get("role", "unknown")] = out.get(user.get("role", "unknown"), 0) + 1
    out["total"] = len(users)
    return out


def storage_status():
    try:
        return {"configured": is_configured(), "container": Config.COSMOS_USERS_CONTAINER,
                "counts": counts()}
    except Exception as exc:
        return {"configured": is_configured(), "container": Config.COSMOS_USERS_CONTAINER,
                "error": str(exc)}


# ---------------------------------------------------------------------------
# Member home location — optional, opt-in, and revocable
# ---------------------------------------------------------------------------

class ProfileError(ValueError):
    pass


def update_member_location(user_id, *, address=None, suburb=None, lat=None, lng=None,
                           share_location=None, alert_radius_km=None):
    """Set or clear a member's home location.

    Location is **optional and opt-in**. `share_location` is the switch the rest
    of the app checks — storing coordinates is not permission to use them, so
    every consumer must test `share_location` rather than merely the presence of
    a latitude. Clearing it (share_location=False) also wipes the coordinates,
    so "stop using my location" means the data is gone, not just ignored.
    """
    user = get_user_raw(user_id)
    if not user:
        raise ProfileError(f"Unknown user {user_id}")
    if user.get("role") != "member":
        raise ProfileError("Only insurance members have a home location.")

    profile = dict(user.get("member_profile") or {})

    if share_location is False:
        profile.update(
            {
                "home_address": None,
                "home_suburb": profile.get("home_suburb"),
                "home_lat": None,
                "home_lng": None,
                "share_location": False,
                "location_updated_at": _now(),
            }
        )
    else:
        if lat is not None and lng is not None:
            try:
                lat, lng = float(lat), float(lng)
            except (TypeError, ValueError):
                raise ProfileError("Latitude and longitude must be numbers.")
            bounds = Config.SA_BOUNDS
            if not (
                bounds["min_lat"] <= lat <= bounds["max_lat"]
                and bounds["min_lng"] <= lng <= bounds["max_lng"]
            ):
                raise ProfileError("That location is outside South Africa.")
            profile["home_lat"] = lat
            profile["home_lng"] = lng
        if address is not None:
            profile["home_address"] = address.strip() or None
        if suburb is not None:
            profile["home_suburb"] = suburb.strip().upper() or None
        if share_location is True:
            if profile.get("home_lat") is None:
                raise ProfileError("Set a home location before turning sharing on.")
            profile["share_location"] = True
        profile["location_updated_at"] = _now()

    if alert_radius_km is not None:
        try:
            radius = float(alert_radius_km)
        except (TypeError, ValueError):
            raise ProfileError("Alert radius must be a number.")
        profile["alert_radius_km"] = max(1.0, min(radius, 50.0))

    user["member_profile"] = profile
    stored = upsert_user(user)
    return _public(stored)


def member_home(user_id):
    """(lat, lng, radius_km) if this member has opted in, else None.

    Consumers must go through here rather than reading the profile directly —
    it's the one place the opt-in is enforced.
    """
    user = get_user_raw(user_id)
    if not user or user.get("role") != "member":
        return None
    profile = user.get("member_profile") or {}
    if not profile.get("share_location"):
        return None
    if profile.get("home_lat") is None or profile.get("home_lng") is None:
        return None
    return (
        profile["home_lat"],
        profile["home_lng"],
        float(profile.get("alert_radius_km") or 10.0),
    )


# ---------------------------------------------------------------------------
# Backwards-compatible views used by the rest of the app
# ---------------------------------------------------------------------------

def _as_member(user):
    profile = user.get("member_profile") or {}
    return {
        "member_id": user["user_id"],
        "name": user.get("full_name"),
        "email": user.get("email"),
        "phone": user.get("phone"),
        "suburb": profile.get("home_suburb"),
        "policy_number": profile.get("policy_number"),
        "home_address": profile.get("home_address"),
        "home_lat": profile.get("home_lat"),
        "home_lng": profile.get("home_lng"),
        "share_location": bool(profile.get("share_location")),
        "alert_radius_km": profile.get("alert_radius_km", 10),
    }


def _as_employee(user):
    profile = user.get("employee_profile") or {}
    return {
        "employee_id": user["user_id"],
        "name": user.get("full_name"),
        "email": user.get("email"),
        "role": profile.get("job_title"),
    }


def _as_unit(user):
    profile = user.get("unit_profile") or {}
    return {
        "unit_id": user["user_id"],
        "name": user.get("full_name"),
        "kind": profile.get("kind"),
        "base_suburb": profile.get("base_suburb"),
        "base_lat": profile.get("base_lat"),
        "base_lng": profile.get("base_lng"),
        "vehicles": profile.get("vehicles", 3),
        "radius_km": profile.get("radius_km", 25),
        "contact": user.get("phone"),
    }


def list_members():
    return [_as_member(u) for u in _load() if u.get("role") == "member"]


def get_member(member_id):
    user = get_user_raw(member_id)
    return _as_member(user) if user and user.get("role") == "member" else None


def list_employees():
    return [_as_employee(u) for u in _load() if u.get("role") == "employee"]


def get_employee(employee_id):
    user = get_user_raw(employee_id)
    return _as_employee(user) if user and user.get("role") == "employee" else None


def list_units():
    return [_as_unit(u) for u in _load() if u.get("role") == "cpu"]


def get_unit(unit_id):
    user = get_user_raw(unit_id)
    return _as_unit(user) if user and user.get("role") == "cpu" else None
