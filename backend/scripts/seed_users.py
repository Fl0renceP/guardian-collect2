"""Seed the Cosmos `users` container: 10 members, 10 employees, 5 CPU companies.

Idempotent — re-running upserts the same documents rather than duplicating, so
it's safe to run whenever you want the demo directory reset.

Usage (from the repo root):
    uv run --with azure-cosmos --with python-dotenv backend/scripts/seed_users.py
    ... backend/scripts/seed_users.py --wipe   # delete users not in this seed

The IDs of the original hardcoded directory (MBR-1001..1003, EMP-201..202,
CPU-301..304) are preserved deliberately: any claim already carrying a
`member_id` or `reviewed_by` keeps resolving after the migration.

Home locations are set for only some members, and `share_location` is False for
several of them — the app has to work for members who decline to share a
location, so the seed data has to include them.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import users_service  # noqa: E402

# Seeded engagement counters behind the Guardian Safety Score. The app doesn't
# yet log sessions or "took the safer route" events, so these stand in for that
# telemetry — see services/member_score_service.py. The camera and claims halves
# of the score are NOT seeded; they read real claim data.
# Deliberately varied so the demo shows a spread of tiers rather than ten
# identical members, including one who has barely engaged.
ACTIVITY = {
    #            app_opens, hotspot_views, alerts_ack, routes_planned, safer_routes, camera_linked
    "MBR-1001": (26, 21, 11, 16, 6, True),
    "MBR-1002": (18, 12, 6, 9, 3, True),
    "MBR-1003": (9, 5, 2, 4, 1, False),
    "MBR-1004": (31, 27, 14, 22, 8, True),
    "MBR-1005": (14, 9, 4, 7, 2, True),
    "MBR-1006": (22, 15, 8, 12, 4, False),
    "MBR-1007": (28, 24, 12, 19, 7, True),
    "MBR-1008": (5, 2, 1, 1, 0, False),
    "MBR-1009": (20, 17, 9, 13, 5, True),
    "MBR-1010": (12, 7, 3, 5, 2, False),
}

MEMBERS = [
    # (id, name, email, phone, policy, suburb, address, lat, lng, share)
    ("MBR-1001", "Thandiwe Nkosi", "thandiwe.nkosi@example.co.za", "+27 82 555 0142",
     "DI-4471-2291", "BRYANSTON", "14 Coleraine Drive, Bryanston", -26.0600, 28.0200, True),
    ("MBR-1002", "Riaan van der Merwe", "riaan.vdm@example.co.za", "+27 83 555 0198",
     "DI-4471-8830", "DURBANVILLE", "8 Wellington Road, Durbanville", -33.8300, 18.6500, True),
    ("MBR-1003", "Ayanda Dlamini", "ayanda.dlamini@example.co.za", "+27 71 555 0077",
     "DI-4471-1046", "MORNINGSIDE", None, None, None, False),
    ("MBR-1004", "Fatima Patel", "fatima.patel@example.co.za", "+27 84 555 0231",
     "DI-4471-3312", "SANDTON", "22 Rivonia Road, Sandton", -26.1076, 28.0567, True),
    ("MBR-1005", "Sizwe Mthembu", "sizwe.mthembu@example.co.za", "+27 79 555 0455",
     "DI-4471-6620", "UMHLANGA", "5 Lagoon Drive, Umhlanga", -29.7264, 31.0836, True),
    ("MBR-1006", "Chantelle Adams", "chantelle.adams@example.co.za", "+27 82 555 0666",
     "DI-4471-7741", "SEA POINT", None, None, None, False),
    ("MBR-1007", "Kagiso Molefe", "kagiso.molefe@example.co.za", "+27 76 555 0912",
     "DI-4471-2288", "CENTURION", "31 Lenchen Avenue, Centurion", -25.8603, 28.1894, True),
    ("MBR-1008", "Nomsa Khumalo", "nomsa.khumalo@example.co.za", "+27 83 555 0388",
     "DI-4471-9903", "SOWETO", None, None, None, False),
    ("MBR-1009", "Deon Fourie", "deon.fourie@example.co.za", "+27 82 555 0517",
     "DI-4471-4455", "STELLENBOSCH", "9 Dorp Street, Stellenbosch", -33.9321, 18.8602, True),
    ("MBR-1010", "Lerato Sithole", "lerato.sithole@example.co.za", "+27 71 555 0740",
     "DI-4471-5567", "ROODEPOORT", None, None, None, False),
]

EMPLOYEES = [
    ("EMP-201", "Sipho Maseko", "sipho.maseko@discovery.co.za", "Claims Assessor",
     ["claims.review"]),
    ("EMP-202", "Lerato Mahlangu", "lerato.mahlangu@discovery.co.za", "Senior Claims Assessor",
     ["claims.review", "claims.escalate"]),
    ("EMP-203", "Johan Botha", "johan.botha@discovery.co.za", "Claims Assessor",
     ["claims.review"]),
    ("EMP-204", "Precious Ndlovu", "precious.ndlovu@discovery.co.za", "Claims Assessor",
     ["claims.review"]),
    ("EMP-205", "Imran Kader", "imran.kader@discovery.co.za", "Fraud Analyst",
     ["claims.review", "claims.flag"]),
    ("EMP-206", "Michelle du Toit", "michelle.dutoit@discovery.co.za", "Claims Team Lead",
     ["claims.review", "claims.escalate", "reports.view"]),
    ("EMP-207", "Tebogo Mokoena", "tebogo.mokoena@discovery.co.za", "Claims Assessor",
     ["claims.review"]),
    ("EMP-208", "Zanele Mabaso", "zanele.mabaso@discovery.co.za", "Data Analyst",
     ["reports.view"]),
    ("EMP-209", "Werner Steyn", "werner.steyn@discovery.co.za", "Claims Assessor",
     ["claims.review"]),
    ("EMP-210", "Nadia Isaacs", "nadia.isaacs@discovery.co.za", "Operations Manager",
     ["claims.review", "reports.view", "units.manage"]),
]

UNITS = [
    # (id, company, kind, base suburb, lat, lng, vehicles, radius, phone, email)
    ("CPU-301", "Sandton Armed Response", "Armed response", "SANDTON",
     -26.1076, 28.0567, 4, 25, "+27 11 555 0300", "ops@sandtonarmed.co.za"),
    ("CPU-302", "Cape Town Metro Response", "Armed response", "CAPE TOWN CITY CENTRE",
     -33.9249, 18.4241, 3, 30, "+27 21 555 0311", "control@ctmetro.co.za"),
    ("CPU-303", "SAPS Pretoria Central", "SAPS", "PRETORIA CENTRAL",
     -25.7479, 28.2293, 5, 30, "+27 12 555 0322", "pretoria.central@saps.gov.za"),
    ("CPU-304", "Durban Coastal Response", "Armed response", "DURBAN CENTRAL",
     -29.8587, 31.0218, 3, 25, "+27 31 555 0333", "ops@durbancoastal.co.za"),
    ("CPU-305", "Midrand Tactical Units", "Armed response", "MIDRAND",
     -25.9895, 28.1284, 4, 20, "+27 11 555 0344", "dispatch@midrandtactical.co.za"),
]


def _base(user_id, role, name, email, phone):
    return {
        "id": user_id,
        "user_id": user_id,
        "role": role,
        "full_name": name,
        "email": email,
        "phone": phone,
        "status": "active",
        # Placeholder only — nothing authenticates yet. Shape is here so the
        # auth work has somewhere to land without a migration.
        "auth": {"provider": None, "password_hash": None, "last_login_at": None},
    }


def build_documents():
    docs = []

    for (uid, name, email, phone, policy, suburb, address, lat, lng, share) in MEMBERS:
        doc = _base(uid, "member", name, email, phone)
        opens, views, acks, planned, safer, camera = ACTIVITY[uid]
        doc["member_profile"] = {
            "policy_number": policy,
            "home_suburb": suburb,
            "home_address": address,
            "home_lat": lat,
            "home_lng": lng,
            # Opt-in. False for members who declined — the app must work for them.
            "share_location": bool(share),
            "alert_radius_km": 10,
            "location_updated_at": None,
            "activity": {
                "app_opens": opens,
                "hotspot_views": views,
                "alerts_acknowledged": acks,
                "routes_planned": planned,
                "safer_routes_taken": safer,
                "camera_linked": camera,
            },
        }
        docs.append(doc)

    for (uid, name, email, title, permissions) in EMPLOYEES:
        doc = _base(uid, "employee", name, email, None)
        doc["employee_profile"] = {
            "employee_number": uid,
            "job_title": title,
            "permissions": permissions,
        }
        docs.append(doc)

    for (uid, company, kind, suburb, lat, lng, vehicles, radius, phone, email) in UNITS:
        doc = _base(uid, "cpu", company, email, phone)
        doc["unit_profile"] = {
            "kind": kind,
            "base_suburb": suburb,
            "base_lat": lat,
            "base_lng": lng,
            "vehicles": vehicles,
            "radius_km": radius,
        }
        docs.append(doc)

    return docs


def main():
    parser = argparse.ArgumentParser(description="Seed the users container.")
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="delete users that aren't part of this seed (does not touch claims)",
    )
    args = parser.parse_args()

    documents = build_documents()
    container = users_service._users_container()

    if args.wipe:
        keep = {d["user_id"] for d in documents}
        existing = list(
            container.query_items("SELECT c.id, c.role FROM c", enable_cross_partition_query=True)
        )
        for item in existing:
            if item["id"] not in keep:
                container.delete_item(item["id"], partition_key=item["role"])
                print(f"  deleted {item['id']}")

    for doc in documents:
        users_service.upsert_user(doc)

    users_service.invalidate()
    counts = users_service.counts()
    print(
        f"Seeded {counts['total']} users -> "
        f"{counts['member']} members, {counts['employee']} employees, {counts['cpu']} units"
    )
    print(f"Container: {container.id} (partition key /role)")

    shared = sum(
        1
        for m in users_service.list_members()
        if m["share_location"]
    )
    print(f"Members sharing a home location: {shared}/{counts['member']} (the rest declined)")


if __name__ == "__main__":
    main()
