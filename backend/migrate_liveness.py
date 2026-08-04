"""Add the liveness column to the detections table.

Liveness is recorded per detection rather than inferred later because it is a
property of the capture, not of the match: the same stored face can be scanned
once from a live person and once from a photo held up to the lens, and the
audit trail has to be able to tell those two events apart afterwards.

NULL means "not reported" — every row written before this migration, and any
client that does not send the field. That is deliberately distinct from FALSE,
which means a client checked and saw no blink.

Usage:
    python backend/migrate_liveness.py
"""

import sys
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Config  # noqa: E402

MIGRATION = """
ALTER TABLE detections
    ADD COLUMN IF NOT EXISTS liveness_confirmed BOOLEAN;

COMMENT ON COLUMN detections.liveness_confirmed IS
    'Client-side blink check at capture time. NULL = not reported by the client.';
"""


def main():
    if not Config.DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured.")

    conn = psycopg2.connect(Config.DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(MIGRATION)
            cur.execute(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'detections' AND column_name = 'liveness_confirmed';
                """
            )
            row = cur.fetchone()
        conn.commit()
        if row:
            print(f"OK: detections.{row[0]} ({row[1]}, nullable={row[2]})")
        else:
            print("FAILED: column not present after migration.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
