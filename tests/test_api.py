"""API tests with FastAPI's TestClient (offline, synthetic stars, tiny trained models)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from transit_hunter.api.main import create_app

Env = tuple[object, Path, Path]


@pytest.fixture()
def client(trained_settings: Env) -> Iterator[TestClient]:
    with TestClient(create_app(warm=False, job_workers=1)) as c:
        yield c


def _wait(client: TestClient, job_id: str, timeout: float = 120.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body: dict[str, Any] = client.get(f"/jobs/{job_id}").json()
        if body["state"] in ("done", "failed"):
            return body
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["offline"] is True
    assert body["model_kind"] == "cnn"


def test_target_search_cache_and_offline(client: TestClient) -> None:
    r = client.get("/targets/search", params={"q": "kic 3"})
    assert r.status_code == 200
    assert r.json()[0] == {
        "query": "kic 3",
        "target_id": "KIC 3",
        "mission": "Kepler",
        "n_products": 0,
        "authors": [],
        "source": "cache",
    }
    r = client.get("/targets/search", params={"q": "Kepler-10"})
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "offline_not_cached"


def test_analyze_job_lifecycle_and_result_cache(client: TestClient) -> None:
    r = client.post("/analyze", json={"target_id": "KIC 1", "max_candidates": 1})
    assert r.status_code == 202
    created = r.json()
    assert created["status_url"] == f"/jobs/{created['job_id']}"
    job = _wait(client, created["job_id"])
    assert job["state"] == "done", job.get("error")
    res = job["result"]
    assert res["target_id"] == "KIC 1"
    cand = res["candidates"][0]
    for key in (
        "period",
        "t0",
        "duration",
        "depth",
        "snr",
        "probabilities",
        "predicted_class",
        "vetting_flags",
        "plot_data",
        "saliency",
    ):
        assert key in cand
    assert set(cand["probabilities"]) == {"pc", "eb", "fp"}
    # Same request again (different spelling) is served from the result cache immediately.
    again = client.post("/analyze", json={"target_id": "kic 1", "max_candidates": 1}).json()
    assert again["state"] == "done"
    cached = client.get(f"/jobs/{again['job_id']}").json()["result"]
    assert cached["source"] == "cache" and cached["cached"] is True
    # force=True bypasses the cache.
    forced = client.post(
        "/analyze", json={"target_id": "KIC 1", "max_candidates": 1, "force": True}
    ).json()
    assert forced["state"] == "queued"
    assert _wait(client, forced["job_id"])["result"]["source"] == "computed"


def test_analyze_errors(client: TestClient) -> None:
    job = _wait(client, client.post("/analyze", json={"target_id": "Kepler-10"}).json()["job_id"])
    assert job["state"] == "failed"
    assert job["error"]["code"] == "offline_not_cached"
    assert client.post("/analyze", json={"target_id": ""}).status_code == 422
    assert client.post("/analyze", json={"target_id": "KIC 1", "mission": "K2"}).status_code == 422
    r = client.get("/jobs/does-not-exist")
    assert r.status_code == 404 and r.json()["detail"]["code"] == "job_not_found"


def test_demo_endpoints_empty(client: TestClient) -> None:
    assert client.get("/demo/targets").json() == []
    r = client.get("/demo/kepler-10")
    assert r.status_code == 404


def test_hunt_job(client: TestClient) -> None:
    r = client.post("/hunt", json={"target_ids": ["KIC 4", "KIC 7", "Kepler-10"]})
    assert r.status_code == 202
    job = _wait(client, r.json()["job_id"], timeout=180)
    assert job["state"] == "done" and job["kind"] == "hunt"
    ranked = job["hunt_result"]["ranked"]
    assert len(ranked) == 3
    assert ranked[-1]["status"] == "offline_not_cached"


def test_openapi_lists_endpoints(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/health", "/targets/search", "/analyze", "/jobs/{job_id}", "/demo/targets", "/hunt"):
        assert p in paths
