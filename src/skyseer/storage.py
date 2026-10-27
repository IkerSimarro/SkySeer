"""The station's record of every night: sessions, events and human reviews (SQLite).

A *session* is one analysed recording or one live night; its folder holds the clips and
images. An *event* stores SkySeer's full report on one detection (``Event.to_dict``) plus
a few indexed columns for fast filtering, and an optional human *review* (the label a
person confirmed or corrected in the dashboard). Reviews are the real-world training data
the learned classifier needs.

SQLite needs no server and is a single file, which suits a Raspberry Pi. WAL mode lets
the live pipeline write while the dashboard reads.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,          -- run folder name
    kind TEXT NOT NULL,                 -- 'file' (a recording) or 'live' (camera)
    source TEXT,                        -- video path or camera id
    started TEXT,                       -- UTC, ISO 8601
    ended TEXT,
    folder TEXT NOT NULL,               -- clips/images paths are relative to this
    meta TEXT NOT NULL DEFAULT '{}'     -- JSON: observer, calibration, frame size, stats
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    event_id INTEGER NOT NULL,          -- the id within its session
    start_time TEXT NOT NULL,
    label TEXT NOT NULL,
    label_source TEXT,
    confidence REAL,
    review TEXT,                        -- a person's verdict: NULL = not reviewed
    data TEXT NOT NULL,                 -- the full event report (JSON)
    UNIQUE (session_id, event_id)
);
CREATE INDEX IF NOT EXISTS events_time ON events (start_time);
CREATE INDEX IF NOT EXISTS events_label ON events (label);
-- Run folders a person removed from SkySeer: never imported again.
CREATE TABLE IF NOT EXISTS removed_runs (folder TEXT PRIMARY KEY);
"""


LABELS = ("Satellite", "Meteor", "Plane", "Junk", "Star")


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)
            columns = {r["name"] for r in db.execute("PRAGMA table_info(sessions)")}
            if "title" not in columns:  # added later: a name a person gave it (rename)
                db.execute("ALTER TABLE sessions ADD COLUMN title TEXT")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:  # commits on success, rolls back on error
                yield db
        finally:
            db.close()

    # ---------------------------------------------------------------- sessions
    def add_session(
        self,
        name: str,
        kind: str,
        folder: str | Path,
        source: str | None = None,
        started: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> int:
        """Create a session, or update the one with this name. Returns its id."""
        with self._lock, self._connect() as db:
            db.execute(
                """INSERT INTO sessions (name, kind, source, started, folder, meta)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT (name) DO UPDATE SET kind=excluded.kind,
                     source=excluded.source, started=excluded.started,
                     folder=excluded.folder, meta=excluded.meta""",
                (name, kind, source, started, str(folder), json.dumps(meta or {})),
            )
            row = db.execute("SELECT id FROM sessions WHERE name = ?", (name,)).fetchone()
            return int(row["id"])

    def finish_session(self, session_id: int, ended: str, meta: dict[str, Any]) -> None:
        with self._lock, self._connect() as db:
            old = db.execute("SELECT meta FROM sessions WHERE id = ?", (session_id,)).fetchone()
            merged = {**json.loads(old["meta"]), **meta} if old else meta
            db.execute(
                "UPDATE sessions SET ended = ?, meta = ? WHERE id = ?",
                (ended, json.dumps(merged), session_id),
            )

    def sessions(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT s.*, COUNT(e.id) AS events,
                     SUM(e.label = 'Satellite') AS satellites,
                     SUM(e.label = 'Meteor') AS meteors,
                     SUM(e.label = 'Plane') AS planes
                   FROM sessions s LEFT JOIN events e
                     ON e.session_id = s.id AND e.label NOT IN ('Junk', 'Star')
                   GROUP BY s.id ORDER BY s.started DESC, s.id DESC"""
            ).fetchall()
        return [_session(r) for r in rows]

    def session(self, session_id: int) -> dict[str, Any] | None:
        return next((s for s in self.sessions() if s["id"] == session_id), None)

    def rename_session(self, session_id: int, title: str | None) -> bool:
        """Give a session a title of its own (None restores its folder name). The name,
        which ties it to its folder, stays."""
        with self._lock, self._connect() as db:
            cur = db.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
            return cur.rowcount > 0

    def delete_session(self, session_id: int, remember: bool = True) -> None:
        """Remove a session's records (its folder on disk is left alone). With
        ``remember`` its folder is never imported again, so it stays removed."""
        with self._lock, self._connect() as db:
            row = db.execute("SELECT folder FROM sessions WHERE id = ?", (session_id,)).fetchone()
            db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            if remember and row is not None:
                db.execute(
                    "INSERT OR IGNORE INTO removed_runs (folder) VALUES (?)",
                    (_folder_key(row["folder"]),),
                )

    def highlights(self, per_session: int = 3) -> dict[int, list[dict[str, Any]]]:
        """Each session's most striking events that have a trail image: fireballs and
        bright meteors, then named satellites and aircraft (long, bright passes first)."""
        with self._connect() as db:
            rows = db.execute(
                """SELECT id, session_id, label,
                     json_extract(data, '$.media.thumbnail') AS thumbnail,
                     json_extract(data, '$.photometry.peak_magnitude') AS magnitude,
                     json_extract(data, '$.duration_s') AS duration,
                     json_extract(data, '$.fireball') AS fireball,
                     COALESCE(json_extract(data, '$.aircraft.name'),
                              json_extract(data, '$.identification.name'),
                              json_extract(data, '$.shower.code'),
                              json_extract(data, '$.shower')) AS name
                   FROM events
                   WHERE label IN ('Satellite', 'Meteor', 'Plane')
                     AND json_extract(data, '$.media.thumbnail') IS NOT NULL"""
            ).fetchall()
        by_session: dict[int, list[dict[str, Any]]] = {}
        for r in sorted(rows, key=_interest, reverse=True):
            picks = by_session.setdefault(r["session_id"], [])
            if len(picks) < per_session:
                picks.append({k: r[k] for k in ("id", "label", "name", "thumbnail", "magnitude")})
        return by_session

    # ------------------------------------------------------------------ events
    def add_events(self, session_id: int, events: list[dict[str, Any]]) -> None:
        """Insert event reports (``Event.to_dict()``); re-adding one replaces it."""
        with self._lock, self._connect() as db:
            db.executemany(
                """INSERT INTO events (session_id, event_id, start_time, label,
                     label_source, confidence, data)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (session_id, event_id) DO UPDATE SET
                     start_time=excluded.start_time, label=excluded.label,
                     label_source=excluded.label_source,
                     confidence=excluded.confidence, data=excluded.data""",
                [
                    (
                        session_id,
                        e["id"],
                        e["start_time"],
                        e["label"],
                        e.get("label_source"),
                        e.get("confidence"),
                        json.dumps(e),
                    )
                    for e in events
                ],
            )

    def add_event(self, session_id: int, event: dict[str, Any]) -> int:
        """Insert (or replace) one event report; returns its database id."""
        self.add_events(session_id, [event])
        with self._connect() as db:
            row = db.execute(
                "SELECT id FROM events WHERE session_id = ? AND event_id = ?",
                (session_id, event["id"]),
            ).fetchone()
        return int(row["id"])

    def events(
        self,
        session_id: int | None = None,
        labels: list[str] | None = None,
        reviewed: bool | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Events, newest first, filtered by session, class, review state and time."""
        where: list[str] = []
        args: list[Any] = []
        if session_id is not None:
            where.append("session_id = ?")
            args.append(session_id)
        if labels:
            where.append(f"label IN ({','.join('?' * len(labels))})")
            args.extend(labels)
        if reviewed is not None:
            where.append("review IS NOT NULL" if reviewed else "review IS NULL")
        if since:
            where.append("start_time >= ?")
            args.append(since)
        if until:
            where.append("start_time < ?")
            args.append(until)
        sql = "SELECT * FROM events"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY start_time DESC, id DESC LIMIT ? OFFSET ?"
        with self._connect() as db:
            rows = db.execute(sql, [*args, limit, offset]).fetchall()
        return [_event(r) for r in rows]

    def event(self, event_id: int) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return _event(row) if row else None

    def review(self, event_id: int, label: str | None) -> bool:
        """Record a person's verdict on an event (``None`` clears it)."""
        if label is not None and label not in LABELS:
            raise ValueError(f"Unknown label {label!r}; expected one of {LABELS}")
        with self._lock, self._connect() as db:
            cur = db.execute("UPDATE events SET review = ? WHERE id = ?", (label, event_id))
            return cur.rowcount > 0

    def reviews(self, session_id: int) -> dict[int, str]:
        """{event id within the session: reviewed label}, for training data."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT event_id, review FROM events WHERE session_id = ? AND review IS NOT NULL",
                (session_id,),
            ).fetchall()
        return {int(r["event_id"]): r["review"] for r in rows}

    # ----------------------------------------------------------------- imports
    def import_run(
        self, folder: str | Path, kind: str = "file", name: str | None = None
    ) -> int | None:
        """Register an analysed run folder (events.json, optional run.json) under ``name``
        (default: the folder's name). Returns the session id, or None without events.json."""
        folder = Path(folder)
        events_file = folder / "events.json"
        if not events_file.exists():
            return None
        info: dict[str, Any] = {}
        if (folder / "run.json").exists():
            info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        events = json.loads(events_file.read_text(encoding="utf-8"))
        started = info.get("started") or (min(e["start_time"] for e in events) if events else None)
        session_id = self.add_session(
            name or folder.name, info.get("kind", kind), folder, info.get("source"), started, info
        )
        with self._lock, self._connect() as db:  # imported on purpose: no longer removed
            db.execute("DELETE FROM removed_runs WHERE folder = ?", (_folder_key(folder),))
        self.add_events(session_id, events)
        if info.get("ended"):
            self.finish_session(session_id, info["ended"], {})
        return session_id

    def import_runs(self, *roots: str | Path) -> int:
        """Register every run folder under ``roots`` not yet known (by folder, so renamed
        sessions aren't imported twice) and not removed by a person. Returns how many."""
        known = {_folder_key(s["folder"]) for s in self.sessions()}
        with self._connect() as db:
            known |= {r["folder"] for r in db.execute("SELECT folder FROM removed_runs")}
        added = 0
        for root in roots:
            for events_file in sorted(Path(root).glob("*/events.json")):
                if _folder_key(events_file.parent) not in known:
                    added += self.import_run(events_file.parent) is not None
        return added


def _folder_key(folder: str | Path) -> str:
    """One spelling per folder (absolute, case-folded on Windows)."""
    return os.path.normcase(os.path.abspath(str(folder)))


def _interest(r: sqlite3.Row) -> float:
    """How striking an event looks: higher first (magnitudes: lower is brighter)."""
    mag, duration = r["magnitude"], r["duration"] or 0.0
    bright = 0.0 if mag is None else 4.0 - mag
    if r["label"] == "Meteor":
        return (100.0 if r["fireball"] else 60.0) + 5 * bright
    if r["label"] == "Satellite":
        return (45.0 if r["name"] else 20.0) + min(duration, 120.0) / 6 + 2 * bright
    return (35.0 if r["name"] else 15.0) + min(duration, 60.0) / 6


def _session(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    out["meta"] = json.loads(out["meta"])
    for key in ("events", "satellites", "meteors", "planes"):
        out[key] = int(out.get(key) or 0)
    return out


def _event(row: sqlite3.Row) -> dict[str, Any]:
    data = json.loads(row["data"])
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "event_id": row["event_id"],
        "review": row["review"],
        **{k: v for k, v in data.items() if k != "id"},
    }
