"""Create the detections event log.

DATA_SCHEMA.md section 2 and PROJECT_CONTEXT.md section 5: every camera check is
recorded, matched or not, in a table separate from the identity registry. That
split is the point — the registry stays small and deliberate (who we knowingly
track) while the log builds the continuous dataset, without one turning into the
other.

Deliberately NOT stored here: the scanned image and its embedding. Logging the
event is operationally useful; keeping a face vector for every stranger who walks
past a camera is the "silently building a biometric database" problem
PROJECT_CONTEXT.md section 5 objected to. If a specific frame needs keeping, that
is an enrolment decision made by a person, via enroll_face.py.

    python migrate_detections.py
"""

import psycopg2

from config import Config

MIGRATE_SQL = """
CREATE TABLE IF NOT EXISTS detections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Where the check came from. 'demo_upload' for the file-upload demo.
    camera_id TEXT NOT NULL DEFAULT 'demo_upload',
    location_lat DOUBLE PRECISION,
    location_lng DOUBLE PRECISION,

    -- ON DELETE SET NULL, not CASCADE: removing someone from the registry must
    -- not erase the history of them having been detected. This is an audit trail.
    matched_person_id UUID REFERENCES persons(id) ON DELETE SET NULL,
    -- Denormalised so the log still reads correctly after a rename, a merge, or
    -- a deletion that nulls the FK above.
    matched_name TEXT,
    match_label TEXT NOT NULL,

    match_score REAL,
    match_threshold REAL,
    margin_to_next REAL,

    -- Records whether the alert CONDITION was met. No push channel is wired up
    -- yet, so this is not proof a human was actually notified.
    alert_sent BOOLEAN NOT NULL DEFAULT FALSE,

    faces_detected INTEGER,
    capture_quality REAL,
    quality_passed BOOLEAN,

    detected_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT detections_label_check CHECK (
        match_label IN ('offender', 'suspect', 'verified', 'no_match', 'no_face')
    )
);

CREATE INDEX IF NOT EXISTS idx_detections_detected_at ON detections(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_detections_person ON detections(matched_person_id);
CREATE INDEX IF NOT EXISTS idx_detections_label ON detections(match_label);
CREATE INDEX IF NOT EXISTS idx_detections_alerts
    ON detections(detected_at DESC) WHERE alert_sent;
"""


def migrate():
    conn = psycopg2.connect(Config.DATABASE_URL)
    cur = conn.cursor()
    try:
        cur.execute(MIGRATE_SQL)
        conn.commit()
        print("detections table ready")

        cur.execute("SELECT count(*) FROM detections")
        print(f"rows: {cur.fetchone()[0]}")

        cur.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'detections'
            ORDER BY ordinal_position;
            """
        )
        print("\ncolumns:")
        for name, dtype in cur.fetchall():
            print(f"  {name:20s} {dtype}")
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    migrate()
