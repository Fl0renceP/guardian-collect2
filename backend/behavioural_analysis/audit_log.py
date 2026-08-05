"""Audit trail — every heuristic trigger, with the inputs it fired on.

A behavioural flag sends a person into a review queue. Weeks later, someone has
to be able to answer "why was this raised?" without the video, which may be
gone. So every trigger is written with the numbers it was computed from, the
thresholds in force at the time, and the resulting scores.

Two sinks, both local for the hackathon:
  * SQLite — queryable, for a reviewer or a dashboard
  * JSONL  — append-only, grep-able, survives a corrupted database file

WHAT IS NOT WRITTEN HERE:
  * No names, member ids, face ids, embeddings or images. The only identifier is
    the anonymous per-run track id, which maps to no person anywhere.
  * No raw pose landmark arrays unless `audit.store_raw_keypoints` is turned on.
    The audit needs the numbers a decision was made from — dwell seconds, crouch
    ratio, speeds — and those are always written. A full 33-point skeleton adds
    nothing to that and is body-geometry data, so it stays out by default.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS behavioural_events (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id               TEXT UNIQUE,
    logged_at              TEXT NOT NULL,   -- when we wrote the row
    timestamp              TEXT NOT NULL,   -- when the behaviour happened
    track_id               TEXT NOT NULL,   -- anonymous, per-run
    location_zone_id       TEXT,
    source                 TEXT,            -- video file or camera
    behavioural_risk_score REAL NOT NULL,
    face_match_confidence  REAL,
    composite_risk_score   REAL NOT NULL,
    requires_human_review  INTEGER NOT NULL,
    triggered_heuristics   TEXT NOT NULL,   -- JSON
    reasoning              TEXT NOT NULL,   -- JSON list of plain-English steps
    config_source          TEXT             -- which config.yaml was in force
);

CREATE TABLE IF NOT EXISTS heuristic_triggers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id    TEXT NOT NULL,
    logged_at   TEXT NOT NULL,
    track_id    TEXT NOT NULL,
    heuristic   TEXT NOT NULL,
    confidence  REAL NOT NULL,
    explanation TEXT NOT NULL,
    inputs      TEXT NOT NULL,              -- JSON: values AND thresholds
    FOREIGN KEY (event_id) REFERENCES behavioural_events(event_id)
);

CREATE INDEX IF NOT EXISTS idx_events_track ON behavioural_events(track_id);
CREATE INDEX IF NOT EXISTS idx_events_review ON behavioural_events(requires_human_review);
CREATE INDEX IF NOT EXISTS idx_triggers_event ON heuristic_triggers(event_id);
"""


class AuditLog:
    """Append-only audit sink. Safe to share across threads."""

    def __init__(
        self,
        *,
        sqlite_path: Optional[Path | str] = None,
        jsonl_path: Optional[Path | str] = None,
        enabled: bool = True,
        base_dir: Optional[Path] = None,
    ):
        self.enabled = enabled
        base = Path(base_dir) if base_dir else Path(__file__).resolve().parent
        self.sqlite_path = self._resolve(sqlite_path, base)
        self.jsonl_path = self._resolve(jsonl_path, base)
        self._lock = threading.Lock()
        self._connection: Optional[sqlite3.Connection] = None

        if self.enabled and self.sqlite_path:
            self._init_db()

    @staticmethod
    def _resolve(path: Optional[Path | str], base: Path) -> Optional[Path]:
        if not path:
            return None
        candidate = Path(path)
        return candidate if candidate.is_absolute() else base / candidate

    def _init_db(self) -> None:
        assert self.sqlite_path is not None
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.sqlite_path), check_same_thread=False)
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def record_event(
        self,
        event: Dict[str, Any],
        *,
        reasoning: Sequence[str],
        source: str = "",
        config_source: str = "",
        heuristic_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Write one event and its per-heuristic inputs.

        `event` is the JSON already built by api_output.build_event, so the audit
        row and the API payload cannot drift apart.
        """
        if not self.enabled:
            return

        logged_at = datetime.now(timezone.utc).isoformat()
        event_id = event.get("event_id") or f"{event.get('track_id')}-{event.get('timestamp')}"
        record = {
            "event_id": event_id,
            "logged_at": logged_at,
            "source": source,
            "config_source": config_source,
            "reasoning": list(reasoning),
            "event": event,
            "heuristic_inputs": heuristic_inputs or {},
        }

        with self._lock:
            self._write_jsonl(record)
            self._write_sqlite(event, event_id, logged_at, source, config_source, reasoning, heuristic_inputs or {})

    def _write_jsonl(self, record: Dict[str, Any]) -> None:
        if not self.jsonl_path:
            return
        try:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        except OSError:
            logger.exception("Could not append to the JSONL audit log at %s", self.jsonl_path)

    def _write_sqlite(
        self,
        event: Dict[str, Any],
        event_id: str,
        logged_at: str,
        source: str,
        config_source: str,
        reasoning: Sequence[str],
        heuristic_inputs: Dict[str, Dict[str, Any]],
    ) -> None:
        if self._connection is None:
            return
        try:
            self._connection.execute(
                """
                INSERT OR IGNORE INTO behavioural_events (
                    event_id, logged_at, timestamp, track_id, location_zone_id, source,
                    behavioural_risk_score, face_match_confidence, composite_risk_score,
                    requires_human_review, triggered_heuristics, reasoning, config_source
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    event_id,
                    logged_at,
                    event.get("timestamp"),
                    event.get("track_id"),
                    event.get("location_zone_id"),
                    source,
                    event.get("behavioural_risk_score"),
                    event.get("face_match_confidence"),
                    event.get("composite_risk_score"),
                    1 if event.get("requires_human_review") else 0,
                    json.dumps(event.get("triggered_heuristics", []), default=str),
                    json.dumps(list(reasoning), default=str),
                    config_source,
                ),
            )

            for trigger in event.get("triggered_heuristics", []):
                heuristic = trigger.get("type")
                self._connection.execute(
                    """
                    INSERT INTO heuristic_triggers (
                        event_id, logged_at, track_id, heuristic, confidence, explanation, inputs
                    ) VALUES (?,?,?,?,?,?,?)
                    """,
                    (
                        event_id,
                        logged_at,
                        event.get("track_id"),
                        heuristic,
                        trigger.get("confidence"),
                        trigger.get("explanation"),
                        json.dumps(heuristic_inputs.get(heuristic, {}), default=str),
                    ),
                )
            self._connection.commit()
        except sqlite3.Error:
            logger.exception("Could not write to the SQLite audit log at %s", self.sqlite_path)

    def recent(self, limit: int = 20, review_only: bool = False) -> List[Dict[str, Any]]:
        """Read back recent events — the review queue, in its simplest form."""
        if self._connection is None:
            return []
        query = "SELECT event_id, timestamp, track_id, location_zone_id, " \
                "behavioural_risk_score, face_match_confidence, composite_risk_score, " \
                "requires_human_review, triggered_heuristics, reasoning FROM behavioural_events"
        if review_only:
            query += " WHERE requires_human_review = 1"
        query += " ORDER BY id DESC LIMIT ?"

        with self._lock:
            rows = self._connection.execute(query, (limit,)).fetchall()

        return [
            {
                "event_id": row[0],
                "timestamp": row[1],
                "track_id": row[2],
                "location_zone_id": row[3],
                "behavioural_risk_score": row[4],
                "face_match_confidence": row[5],
                "composite_risk_score": row[6],
                "requires_human_review": bool(row[7]),
                "triggered_heuristics": json.loads(row[8]) if row[8] else [],
                "reasoning": json.loads(row[9]) if row[9] else [],
            }
            for row in rows
        ]

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None


def from_settings(settings, *, base_dir: Optional[Path] = None) -> AuditLog:
    """Build an AuditLog from the `audit:` block of config.yaml."""
    cfg = settings.audit
    return AuditLog(
        sqlite_path=cfg.get("sqlite_path"),
        jsonl_path=cfg.get("jsonl_path"),
        enabled=bool(cfg.get("enabled", True)),
        base_dir=base_dir,
    )
