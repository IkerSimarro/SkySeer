"""The live station, run on a replayed synthetic video (no camera needed)."""

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from skyseer.api import create_app
from skyseer.config import load_config
from skyseer.live import LiveStation, close_interrupted
from skyseer.sources import ReplaySource
from skyseer.storage import Store


def _station(path, tmp_path, speed=0.0):
    store = Store(tmp_path / "db.sqlite")
    source = ReplaySource(path, speed=speed)
    return LiveStation(source, load_config(), store, "replay", tmp_path / "night"), store


def test_station_stores_every_event_and_its_folder(synth_clip, tmp_path):
    path, truth = synth_clip
    station, store = _station(path, tmp_path)
    station.start()
    station.wait()
    status = station.status()
    assert not status["running"] and status["error"] is None
    assert status["frames"] == 16 * 30
    session = store.sessions()[0]
    assert session["kind"] == "live" and session["ended"]
    # Ends at its last frame's time (a replay's own clock), not the computer's: 16 s.
    from datetime import datetime

    length = datetime.fromisoformat(session["ended"]) - datetime.fromisoformat(session["started"])
    assert abs(length.total_seconds() - 479 / 30) < 0.5
    assert session["events"] == len(truth.objects)
    folder = tmp_path / "night"
    events = json.loads((folder / "events.json").read_text())
    real = [e for e in events if e["label"] not in ("Junk", "Star")]
    assert len(real) == len(truth.objects)
    assert all((folder / e["media"]["clip"]).exists() for e in real)
    assert len((folder / "events.jsonl").read_text().splitlines()) == len(events)
    assert json.loads((folder / "run.json").read_text())["frames"] == 16 * 30
    assert station.preview_jpeg()[:2] == b"\xff\xd8"  # a JPEG
    # The folder is self-contained: another station database can import it.
    other = Store(tmp_path / "other.sqlite")
    assert other.import_run(folder) is not None
    assert other.sessions()[0]["events"] == len(truth.objects)


def test_dashboards_receive_events_as_they_happen(synth_clip, tmp_path):
    path, truth = synth_clip
    station, _ = _station(path, tmp_path, speed=8.0)  # 16 s of video in ~2 s

    def on_disk(url):  # /media/<session>/clips/0001.jpg -> the file in the night's folder
        return (tmp_path / "night" / url.split("/", 3)[3]).exists()

    async def listen():
        got, ready = [], []
        updates = station.updates()
        first = await updates.__anext__()
        station.start()
        while station.running or not got:
            msg = await asyncio.wait_for(updates.__anext__(), timeout=10)
            if msg["type"] == "event":
                got.append(msg["event"])
                media = msg["event"]["media"]
                ready.append(on_disk(media["thumbnail"]) and on_disk(media["clip"]))
            if len(got) == len(truth.objects):
                break
        await updates.aclose()
        return first, got, ready

    first, got, ready = asyncio.run(listen())
    station.wait()
    assert first["type"] == "status"
    assert len(got) == len(truth.objects)
    assert all(e["media"]["clip"].startswith("/media/") for e in got)
    # Each event is announced only once its clip and trail image exist, so a dashboard
    # never shows a broken image (found while recording the README demo).
    assert all(ready)


def test_replay_runs_in_real_time(synth_clip):
    path, _ = synth_clip
    source = ReplaySource(path, speed=4.0)
    t0 = time.perf_counter()
    for frame in source:
        if frame.index == 119:  # 4 s of video at 4x speed
            break
    assert 0.9 < time.perf_counter() - t0 < 1.6
    source.close()


def test_live_api_without_a_camera(tmp_path):
    client = TestClient(create_app(load_config(), Store(tmp_path / "db.sqlite"), static_dir=None))
    with client.websocket_connect("/ws/live") as ws:
        assert ws.receive_json() == {"type": "status", "live": None}
    assert client.get("/api/live/preview.mjpg").status_code == 404


def test_a_real_server_streams_events_over_websocket(synth_clip, tmp_path):
    """Through uvicorn, as `skyseer serve` runs it (the test client alone can't catch a
    missing WebSocket library, which once broke the Live page)."""
    import socket
    import threading

    import uvicorn
    from websockets.sync.client import connect

    path, _ = synth_clip
    station, store = _station(path, tmp_path, speed=4.0)
    app = create_app(load_config(), store, station, static_dir=None)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    station.start()
    try:
        with connect(f"ws://127.0.0.1:{port}/ws/live") as ws:
            first = json.loads(ws.recv(timeout=10))
            assert first["type"] == "status" and first["live"]["camera"] == "replay"
            kinds = set()
            while "event" not in kinds:
                kinds.add(json.loads(ws.recv(timeout=10))["type"])
    finally:
        station.stop()
        server.should_exit = True
        thread.join(10)


def test_a_night_cut_off_by_a_power_cut_ends_at_its_last_frame(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    beat = datetime(2026, 12, 12, 23, 0, tzinfo=UTC)
    folder = tmp_path / "cut"
    folder.mkdir()
    cut = store.add_session(
        "cut",
        "live",
        folder,
        "0",
        "2026-12-12T18:30:00+00:00",
        {"heartbeat": beat.isoformat(), "last_frame": "2026-12-12T22:59:58+00:00"},
    )
    # An older session from before heartbeats: it ends at its last event.
    old = store.add_session("old", "live", tmp_path / "gone", "0", "2026-12-10T18:30:00+00:00")
    event = {"id": 1, "start_time": "2026-12-10T20:00:00+00:00", "label": "Meteor"}
    store.add_event(old, event | {"end_time": "2026-12-10T20:00:01+00:00"})
    # Still beating 2 minutes ago: running (perhaps in another process), left alone.
    alive = store.add_session(
        "alive",
        "live",
        tmp_path / "alive",
        "0",
        "2026-12-12T18:30:00+00:00",
        {"heartbeat": (beat + timedelta(minutes=8)).isoformat()},
    )

    assert sorted(close_interrupted(store, beat + timedelta(minutes=10))) == [cut, old]
    s = store.session(cut)
    assert s["ended"] == "2026-12-12T22:59:58+00:00" and s["meta"]["interrupted"] is True
    assert json.loads((folder / "run.json").read_text())["interrupted"] is True
    assert (folder / "events.json").exists()
    assert store.session(old)["ended"] == "2026-12-10T20:00:01+00:00"
    assert store.session(alive)["ended"] is None
    assert close_interrupted(store, beat + timedelta(minutes=10)) == []  # once only


def test_a_running_station_records_a_heartbeat(synth_clip, tmp_path):
    path, _ = synth_clip
    station, store = _station(path, tmp_path)
    station.start()
    station.wait()
    meta = store.sessions()[0]["meta"]
    assert meta["last_frame"] and meta["heartbeat"] and meta["frames"] >= 1
    assert "interrupted" not in meta
