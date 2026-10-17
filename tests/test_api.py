"""The station database and web API, on a synthetic scene analysed by the real CLI."""

import csv
import io
import json

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from skyseer.api import create_app
from skyseer.cli import app as cli
from skyseer.config import load_config
from skyseer.storage import Store


@pytest.fixture(scope="module")
def analysed(synth_clip, tmp_path_factory):
    """A run folder from `skyseer analyze` (events.json, run.json, clips)."""
    path, truth = synth_clip
    out = tmp_path_factory.mktemp("run") / "night1"
    result = CliRunner().invoke(cli, ["analyze", str(path), "-o", str(out), "--no-video"])
    assert result.exit_code == 0, result.output
    return out, truth


@pytest.fixture
def client(analysed, tmp_path):
    store = Store(tmp_path / "test.db")
    assert store.import_run(analysed[0]) is not None
    return TestClient(create_app(load_config(), store, static_dir=None))


def test_sessions_list_counts(client, analysed):
    _, truth = analysed
    sessions = client.get("/api/sessions").json()
    assert len(sessions) == 1
    s = sessions[0]
    assert s["name"] == "night1" and s["kind"] == "file"
    assert s["events"] == len(truth.objects)
    assert s["satellites"] + s["meteors"] + s["planes"] == s["events"]
    assert s["meta"]["fps"] == 30.0 and s["meta"]["frame_size"] == [640, 360]


def test_events_list_detail_and_filters(client, analysed):
    _, truth = analysed
    events = client.get("/api/events").json()
    assert len(events) == len(truth.objects)
    assert events == sorted(events, key=lambda e: e["start_time"], reverse=True)
    meteors = client.get("/api/events", params={"label": "Meteor"}).json()
    assert meteors and all(e["label"] == "Meteor" for e in meteors)
    detail = client.get(f"/api/events/{events[0]['id']}").json()
    assert detail["track"] and detail["features"]  # the full report
    assert "track" not in events[0]  # lists stay light
    assert client.get("/api/events/99999").status_code == 404


def test_media_is_served_and_confined_to_the_run(client):
    e = client.get("/api/events").json()[0]
    for kind in ("clip", "thumbnail", "context"):
        r = client.get(e["media"][kind])
        assert r.status_code == 200 and len(r.content) > 1000
    sid = e["session_id"]
    assert client.get(f"/media/{sid}/../../pyproject.toml").status_code == 404
    assert client.get(f"/media/{sid}/%2e%2e/%2e%2e/pyproject.toml").status_code == 404


def test_reviews_are_stored(client):
    e = client.get("/api/events").json()[0]
    url = f"/api/events/{e['id']}/review"
    assert client.put(url, json={"label": "Junk"}).json()["review"] == "Junk"
    assert client.get(f"/api/events/{e['id']}").json()["review"] == "Junk"
    assert client.get("/api/events", params={"reviewed": True}).json()[0]["id"] == e["id"]
    assert client.put(url, json={"label": "UFO"}).status_code == 400
    assert client.put(url, json={"label": None}).json()["review"] is None


def test_night_report(client, analysed):
    _, truth = analysed
    sid = client.get("/api/sessions").json()[0]["id"]
    report = client.get(f"/api/sessions/{sid}/report").json()
    assert sum(report["counts"].values()) == len(truth.objects)
    bins = report["timeline"]["bins"]
    assert sum(b["Satellite"] + b["Meteor"] + b["Plane"] for b in bins) == len(truth.objects)
    assert len(report["meteors"]) == report["counts"]["Meteor"]
    assert 15 < report["duration_s"] < 17  # the 16-second clip


def test_exports(client, analysed):
    _, truth = analysed
    sid = client.get("/api/sessions").json()[0]["id"]
    r = client.get(f"/api/sessions/{sid}/export.csv")
    assert r.headers["content-type"].startswith("text/csv")
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert len(rows) == len(truth.objects) and "peak_mag" in rows[0]
    data = client.get(f"/api/sessions/{sid}/export.json").json()
    assert len(data["events"]) == len(truth.objects)


def test_health(client):
    h = client.get("/api/health").json()
    assert {"cpu_percent", "disk_free_gb", "uptime_s", "version"} <= set(h)
    assert h["live"] is None


def test_runs_are_imported_once(analysed, tmp_path):
    store = Store(tmp_path / "db.sqlite")
    assert store.import_runs(analysed[0].parent) == 1
    assert store.import_runs(analysed[0].parent) == 0
    sid = store.sessions()[0]["id"]
    store.delete_session(sid)
    assert store.sessions() == [] and store.events() == []


def test_reviews_become_training_rows(analysed, tmp_path):
    from skyseer.classify.dataset import arrays, station_rows

    store = Store(tmp_path / "db.sqlite")
    sid = store.import_run(analysed[0])
    e = store.events(session_id=sid)[0]
    store.review(e["id"], "Junk")
    assert station_rows(store, load_config()) == []  # a simulation: not real training data
    s = store.session(sid)
    meta = {**s["meta"], "simulated": False}  # pretend it was a real night
    store.add_session(s["name"], s["kind"], s["folder"], s["source"], s["started"], meta)
    rows = station_rows(store, load_config())
    assert len(rows) == 1 and rows[0]["label"] == "Junk" and rows[0]["kind"] == "review"
    X, y, groups, _ = arrays(rows)
    assert X.shape[0] == 1 and y[0] == "Junk" and groups[0] == "station-night1"


def test_uploaded_videos_are_analysed_in_the_background(synth_clip, tmp_path):
    import time

    path, truth = synth_clip
    storage = {"database": str(tmp_path / "db.sqlite"), "runs_dir": str(tmp_path / "runs")}
    cfg = load_config(storage=storage)
    store = Store(cfg.storage.database)
    client = TestClient(create_app(cfg, store, static_dir=None))
    with open(path, "rb") as fh:
        r = client.post(
            "/api/analyze",
            files={"file": ("night.mp4", fh, "video/mp4")},
            data={"start_time": "2026-10-01T22:00:00", "sky": "false"},
        )
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    deadline = time.time() + 120
    while (job := client.get(f"/api/jobs/{job_id}").json())["state"] not in ("done", "failed"):
        assert time.time() < deadline
        time.sleep(0.5)
    assert job["state"] == "done", job["error"]
    session = client.get(f"/api/sessions/{job['session_id']}").json()
    assert session["events"] == len(truth.objects)
    assert session["started"].startswith("2026-10-01T22:00:00")
    assert (tmp_path / "uploads").is_dir()


def test_uploads_must_be_videos(client):
    r = client.post("/api/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400
    r = client.post(
        "/api/analyze",
        files={"file": ("a.mp4", b"x", "video/mp4")},
        data={"start_time": "yesterday"},
    )
    assert r.status_code == 400


def test_an_unbuilt_dashboard_explains_how_to_build_it(tmp_path):
    """A fresh clone has no compiled dashboard (it is git-ignored): say how to build it
    instead of answering 404."""
    store = Store(tmp_path / "db.sqlite")
    client = TestClient(create_app(load_config(), store, static_dir=tmp_path / "missing"))
    r = client.get("/")
    assert r.status_code == 200 and "npm run build" in r.text
    assert client.get("/api/health").status_code == 200


def test_clips_without_markings(client, analysed, tmp_path):
    """Meteors have one at once; for any other event of an analysed recording it is made
    on request from the original video; a live night can't make one afterwards."""
    from skyseer.clips import clean_path

    from .test_clips import coloured_pixels

    folder, _ = analysed
    events = client.get("/api/events").json()
    meteor = next(e for e in events if e["label"] == "Meteor")
    other = next(e for e in events if e["label"] != "Meteor")
    media = client.get(f"/api/events/{meteor['id']}").json()["media"]
    assert media["clean_status"] == "ready" and media["clean"].endswith("_clean.mp4")
    assert client.get(media["clean"]).status_code == 200

    media = client.get(f"/api/events/{other['id']}").json()["media"]
    assert media["clean_status"] == "on_request" and "clean" not in media
    made = client.post(f"/api/events/{other['id']}/clean-clip").json()
    assert made["clean_status"] == "ready"
    clean_file = folder / made["clean"].split("/", 3)[3]
    n_clean, coloured = coloured_pixels(clean_file)
    n_marked, _ = coloured_pixels(folder / made["clip"].split("/", 3)[3])
    assert n_clean == n_marked and coloured == 0
    stamp = clean_file.stat().st_mtime_ns
    assert client.post(f"/api/events/{other['id']}/clean-clip").status_code == 200
    assert clean_file.stat().st_mtime_ns == stamp  # made once, then reused

    # A live night, or a recording whose video has gone: refused with the reason.
    store = Store(tmp_path / "other.db")
    run_events = json.loads((folder / "events.json").read_text())
    why = {}
    for kind, source, reason in (("live", None, "Live nights"), ("file", "gone.mp4", "no longer")):
        sid = store.add_session(f"{kind}-night", kind, folder, source, None, {})
        store.add_events(sid, run_events)
        why[sid] = reason
    other_client = TestClient(create_app(load_config(), store, static_dir=None))
    refused = 0
    for e in other_client.get("/api/events").json():
        detail = other_client.get(f"/api/events/{e['id']}").json()["media"]
        if (folder / clean_path(detail["clip"].split("/", 3)[3])).exists():
            continue  # a clean copy already exists (meteors, and the one made above)
        assert detail["clean_status"] == "unavailable"
        assert why[e["session_id"]] in detail["clean_reason"]
        r = other_client.post(f"/api/events/{e['id']}/clean-clip")
        assert r.status_code == 409 and r.json()["detail"] == detail["clean_reason"]
        refused += 1
    assert refused >= 2
