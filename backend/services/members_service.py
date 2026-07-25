"""Demo member directory.

**This is not authentication.** The hackathon build has no login, so the member
identity is chosen in the UI and trusted by the API. Every endpoint that takes a
`member_id` would need a real authenticated principal before this goes anywhere
near production — see the note in README under "Known gaps".
"""

MEMBERS = [
    {
        "member_id": "MBR-1001",
        "name": "Thandiwe Nkosi",
        "email": "thandiwe.nkosi@example.co.za",
        "phone": "+27 82 555 0142",
        "suburb": "BRYANSTON",
        "policy_number": "DI-4471-2291",
    },
    {
        "member_id": "MBR-1002",
        "name": "Riaan van der Merwe",
        "email": "riaan.vdm@example.co.za",
        "phone": "+27 83 555 0198",
        "suburb": "DURBANVILLE",
        "policy_number": "DI-4471-8830",
    },
    {
        "member_id": "MBR-1003",
        "name": "Ayanda Dlamini",
        "email": "ayanda.dlamini@example.co.za",
        "phone": "+27 71 555 0077",
        "suburb": "MORNINGSIDE",
        "policy_number": "DI-4471-1046",
    },
]

EMPLOYEES = [
    {"employee_id": "EMP-201", "name": "Sipho Maseko", "role": "Claims Assessor"},
    {"employee_id": "EMP-202", "name": "Lerato Mahlangu", "role": "Senior Claims Assessor"},
]

_BY_ID = {m["member_id"]: m for m in MEMBERS}
_EMP_BY_ID = {e["employee_id"]: e for e in EMPLOYEES}


def list_members():
    return MEMBERS


def get_member(member_id):
    return _BY_ID.get(member_id)


def list_employees():
    return EMPLOYEES


def get_employee(employee_id):
    return _EMP_BY_ID.get(employee_id)
