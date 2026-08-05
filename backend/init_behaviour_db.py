"""Create the behavioural event and review tables.

    python init_behaviour_db.py

UNLIKE init_db.py AND init_plate_db.py, THIS SCRIPT DOES NOT DROP ANYTHING.
Those scripts drop and recreate because they reseed fixed demo data, so losing
the old rows is the point. These two tables hold event history and human review
decisions — a reviewer's confirm/deny is the audit trail's only signature, and a
setup script must never be able to erase it. Re-running this is a no-op.
"""

import logging
import sys

import psycopg2

from config import Config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CREATE_BEHAVIOUR_SCHEMA_SQL = """
-- Every behavioural event, flagged or not.
--
-- Events BELOW the review threshold are stored too, on purpose: they are the
-- denominator. Without them there is no way to say what proportion of flags
-- turned out to be false, which is the only honest measure of whether fusing
-- behaviour with facial recognition reduces false positives at all.
CREATE TABLE IF NOT EXISTS behavioural_events (
    event_id               TEXT PRIMARY KEY,
    -- Anonymous, per-run track id from the behavioural module. It maps to no
    -- person, no member and nothing in the face registry. The ONLY link to an
    -- identity anywhere in this schema is behavioural_reviews.matched_person_id,
    -- and a human has to put it there.
    track_id               TEXT NOT NULL,
    camera_id              TEXT,
    location_zone_id       TEXT,
    -- No PostGIS in this database (only pgvector), so coordinates are plain
    -- doubles, matching the detections table rather than inventing a geometry
    -- column the extension could not index.
    location_lat           DOUBLE PRECISION,
    location_lng           DOUBLE PRECISION,
    occurred_at            TIMESTAMP WITH TIME ZONE NOT NULL,
    received_at            TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    behavioural_risk_score REAL NOT NULL,
    face_match_confidence  REAL,
    composite_risk_score   REAL NOT NULL,
    requires_human_review  BOOLEAN NOT NULL DEFAULT FALSE,
    -- Each entry carries {type, confidence, explanation}. The explanation is
    -- mandatory and enforced in the service: a score with no sentence a human
    -- can read is not reviewable, which is this whole module's justification.
    triggered_heuristics   JSONB NOT NULL,
    reasoning              JSONB,
    source                 TEXT
);

CREATE INDEX IF NOT EXISTS idx_behavioural_events_camera
    ON behavioural_events(camera_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_behavioural_events_review
    ON behavioural_events(requires_human_review, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_behavioural_events_track
    ON behavioural_events(track_id);

-- Human review decisions. One row per event that reached a person.
CREATE SEQUENCE IF NOT EXISTS behavioural_review_seq;

CREATE TABLE IF NOT EXISTS behavioural_reviews (
    review_id         TEXT PRIMARY KEY
                      DEFAULT ('rev-' || lpad(nextval('behavioural_review_seq')::text, 6, '0')),
    event_id          TEXT NOT NULL UNIQUE
                      REFERENCES behavioural_events(event_id) ON DELETE CASCADE,
    status            TEXT NOT NULL DEFAULT 'pending'
                      CHECK (status IN ('pending', 'confirmed', 'denied')),
    -- The one and only identity link in this schema, and it is nullable
    -- because most flags never have one. A human writes it by confirming;
    -- nothing automatic may ever set it (PROJECT_CONTEXT section 5).
    matched_person_id UUID REFERENCES persons(id),
    matched_label     person_status,
    opened_at         TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    decided_by        TEXT,
    decided_at        TIMESTAMP WITH TIME ZONE,
    decision_note     TEXT,
    -- Required when status = 'denied'. A denied flag is a measured false
    -- positive and the reason is the most valuable data this system produces.
    denial_reason     TEXT,
    clip_url          TEXT
);

CREATE INDEX IF NOT EXISTS idx_behavioural_reviews_status
    ON behavioural_reviews(status, opened_at DESC);

-- Every decision ever taken on a review, append-only.
--
-- behavioural_reviews holds the CURRENT state; this holds the HISTORY, and the
-- two serve different purposes. A reviewer confirming, then reopening, then
-- denying is three facts, not one — and the middle one is exactly what an
-- audit needs to see. Nothing in here is ever updated or deleted; a reversal
-- is a new row saying so, not the erasure of the row it reverses.
CREATE TABLE IF NOT EXISTS behavioural_review_decisions (
    id           BIGSERIAL PRIMARY KEY,
    review_id    TEXT NOT NULL REFERENCES behavioural_reviews(review_id) ON DELETE CASCADE,
    decision     TEXT NOT NULL CHECK (decision IN ('confirm', 'deny', 'reopen')),
    -- WHO. Supplied by the client and trusted, because this app has no
    -- authentication (PROJECT_CONTEXT section 9). On an identification decision
    -- this field is the audit trail's only signature, so it has to become real
    -- before the feature does.
    reviewer_id  TEXT NOT NULL,
    decided_at   TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- What the reviewer asserted about identity, when there was one to assert.
    person_id    UUID REFERENCES persons(id),
    label        person_status,
    -- Required on a deny: a denied flag is a measured false positive and the
    -- reason is the most valuable data this system produces.
    reason       TEXT,
    note         TEXT,
    -- Whether this decision propagated to the identity registry. Recorded
    -- because "did this change persons.status" must be answerable later.
    identity_written BOOLEAN NOT NULL DEFAULT FALSE,
    alerts_sent  TEXT
);

CREATE INDEX IF NOT EXISTS idx_behavioural_decisions_review
    ON behavioural_review_decisions(review_id, decided_at DESC);
CREATE INDEX IF NOT EXISTS idx_behavioural_decisions_reviewer
    ON behavioural_review_decisions(reviewer_id, decided_at DESC);

-- Where each tracked BODY was, moment to moment.
--
-- This exists solely so a face match can be attached to a body. A face scan
-- knows where the FACE was; only the behavioural pipeline knows where the
-- BODIES were, and without both there is no way to say whose face it is once
-- two people are in shot.
--
-- These rows are deliberately short-lived (see PRESENCE_RETENTION_MINUTES in
-- services/behaviour_track_service.py). A continuous record of where every
-- body stood is far more intrusive than the sparse events it exists to support,
-- and it has no value once the correlation window has passed.
CREATE TABLE IF NOT EXISTS behavioural_tracks (
    id          BIGSERIAL PRIMARY KEY,
    camera_id   TEXT NOT NULL,
    track_id    TEXT NOT NULL,
    observed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    -- Normalised 0..1 against the frame, so the box survives every resolution
    -- change between the camera, the browser and the analysis.
    bbox_x      REAL NOT NULL,
    bbox_y      REAL NOT NULL,
    bbox_w      REAL NOT NULL,
    bbox_h      REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_behavioural_tracks_lookup
    ON behavioural_tracks(camera_id, observed_at DESC);

-- The result of a successful join: this anonymous track, on this camera, is
-- believed to be this person.
--
-- IT IS A HYPOTHESIS, NOT AN IDENTIFICATION. It is produced by geometry — a
-- face box centre falling inside a body box at roughly the same moment — and
-- the evidence for it is stored alongside so a reviewer can judge it rather
-- than take it. A human's decision (behavioural_reviews.matched_person_id)
-- always outranks a row in this table.
CREATE TABLE IF NOT EXISTS behavioural_face_links (
    camera_id        TEXT NOT NULL,
    track_id         TEXT NOT NULL,
    person_id        UUID REFERENCES persons(id) ON DELETE CASCADE,
    label            person_status,
    confidence       REAL,
    match_distance   REAL,
    scan_captured_at TIMESTAMP WITH TIME ZONE,
    linked_at        TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    -- Evidence for the join itself, kept so it can be second-guessed.
    time_delta_ms    INTEGER,
    candidates       INTEGER,
    PRIMARY KEY (camera_id, track_id)
);
"""


def main():
    if not Config.DATABASE_URL:
        logger.error("DATABASE_URL is not set. Check backend/.env")
        return 1

    try:
        connection = psycopg2.connect(
            Config.DATABASE_URL, connect_timeout=Config.DB_CONNECT_TIMEOUT_SECONDS
        )
    except psycopg2.OperationalError as exc:
        logger.error("Could not connect to the database: %s", exc)
        return 1

    try:
        with connection, connection.cursor() as cursor:
            cursor.execute(CREATE_BEHAVIOUR_SCHEMA_SQL)
            cursor.execute(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name IN ('behavioural_events', 'behavioural_reviews')
                ORDER BY table_name
                """
            )
            created = [row[0] for row in cursor.fetchall()]
        logger.info("Behavioural tables ready: %s", ", ".join(created))
        return 0
    except psycopg2.Error as exc:
        logger.error("Schema creation failed: %s", exc)
        return 1
    finally:
        connection.close()


if __name__ == "__main__":
    sys.exit(main())
