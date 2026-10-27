"""Dashboard analyses: their own low-priority process, stop and resume, kept across restarts."""

import time

import pytest
from fastapi.testclient import TestClient

from skyseer.api import create_app
from skyseer.api.jobs import Jobs
from skyseer.api.worker import threads_for
from skyseer.config import load_config
from skyseer.storage import Store


@pytest.fixture
def setup(tmp_path):
    storage = {"database": str(tmp_path / "db.sqlite"), "runs_dir": str(tmp_path / "runs")}
    cfg = load_config(storage=storage)
    return cfg, Store(cfg.storage.database)


def _upload(client, path, **data):
    with open(path, "rb") as fh:
        r = client.post(
            "/api/analyze",
            files={"file": ("night.mp4", fh, "video/mp4")},
            data={"start_time": "2026-10-01T22:00:00", "sky": "false", **data},
        )
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _wait(client, job_id, states, timeout=120):
    deadline = time.time() + timeout
    while (job := client.get(f"/api/jobs/{job_id}").json())["state"] not in states:
        assert time.time() < deadline, job
        time.sleep(0.1)
    return job


def test_stop_and_resume(synth_clip, setup):
    path, truth = synth_clip
    cfg, store = setup
    with TestClient(create_app(cfg, store, static_dir=None)) as client:
        first = _upload(client, path)
        second = _upload(client, path, speed="quick")  # waits for the first
        queued = client.get(f"/api/jobs/{second}").json()
        assert queued["state"] == "queued" and queued["speed"] == "quick"
        _wait(client, first, ("running",))
        stopped = client.post(f"/api/jobs/{first}/stop").json()  # while the worker starts
        assert stopped["state"] == "stopped" and stopped["resumable"]
        assert client.post(f"/api/jobs/{second}/stop").json()["state"] == "stopped"
        assert client.delete(f"/api/jobs/{second}").json() == {"removed": True}
        assert [j["id"] for j in client.get("/api/jobs").json()] == [first]

        assert client.post(f"/api/jobs/{first}/resume").json()["state"] == "queued"
        job = _wait(client, first, ("done", "failed"))
        assert job["state"] == "done", job["error"]
        assert job["speed"] == "full"
        session = client.get(f"/api/sessions/{job['session_id']}").json()
        assert session["events"] == len(truth.objects)
        assert client.post(f"/api/jobs/{first}/resume").status_code == 409  # done: no


def test_unknown_speed_is_refused(synth_clip, setup):
    cfg, store = setup
    client = TestClient(create_app(cfg, store, static_dir=None))
    with open(synth_clip[0], "rb") as fh:
        r = client.post(
            "/api/analyze", files={"file": ("a.mp4", fh, "video/mp4")}, data={"speed": "warp"}
        )
    assert r.status_code == 400


def test_analyses_survive_a_restart(setup):
    """An analysis running when the server closed is listed as stopped, ready to resume."""
    cfg, store = setup
    base = {"progress": 0.4, "created": "2026-10-27T10:00:00+00:00", "out_dir": "nowhere"}
    store.save_job({**base, "id": "a", "state": "running"})
    store.save_job({**base, "id": "b", "state": "done", "progress": 1.0})
    jobs = {j["id"]: j for j in Jobs(cfg, store).all()}
    assert jobs["a"]["state"] == "stopped" and jobs["a"]["resumable"]
    assert jobs["b"]["state"] == "done" and not jobs["b"]["resumable"]
    assert "out_dir" not in jobs["a"]  # paths stay on the server
    assert {j["id"]: j["state"] for j in store.jobs()} == {"a": "stopped", "b": "done"}


def test_analyses_leave_half_the_cores_free():
    assert threads_for(8) == 4
    assert threads_for(1) == 1
