"""Compatibility shim over `users_service`.

The member / employee / unit directory used to be hardcoded Python lists here.
It now lives in the Cosmos `users` container (see `users_service`), but the rest
of the app still imports these names, so they're re-exported rather than
scattering an edit across every caller.

**Prefer `users_service` in new code.** This module exists so the migration
didn't have to touch `claim_routes` and `cpu_routes` in the same change.

Still not authentication: whichever identity the UI sends is trusted.
"""

from services.users_service import (  # noqa: F401
    get_employee,
    get_member,
    get_unit,
    list_employees,
    list_members,
    list_units,
)

__all__ = [
    "list_members",
    "get_member",
    "list_employees",
    "get_employee",
    "list_units",
    "get_unit",
]
