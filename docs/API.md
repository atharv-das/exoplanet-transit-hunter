# API

Base URL: `http://localhost:8000` (`make api`). Interactive docs: `/docs` (Swagger) and `/redoc`.
Responses over 2 KB are gzip-compressed. Every error body has the same shape:

```json
{"detail": {"code": "target_not_found", "message": "no Kepler/TESS light curves found for 'Foo-1'"}}
```

| Code | Meaning | Typical HTTP |
|---|---|---|
| `target_not_found` | MAST has no Kepler/TESS light curves for the query | 404 / job failed |
| `offline_not_cached` | Server runs with `TH_OFFLINE=true` and the target is not cached | 404 / job failed |
| `no_data` | Products exist but contain no usable flux | job failed |
| `lightcurve_too_short` | Too few valid points after cleaning | job failed |
| `search_failed` | Baseline too short for the period range, or nothing could be folded | job failed |
| `job_not_found` | Unknown job ID | 404 |
| `internal_error` | Unexpected exception (logged server-side) | job failed / 502 |

"No significant signal" is **not** an error: the job succeeds with `status: "no_significant_signal"`.

---

## `GET /health`

```json
{"status": "ok", "version": "0.1.0", "model_kind": "cnn", "offline": false, "demo_targets": 6, "jobs_running": 0}
```

## `GET /targets/search?q=<name or id>&mission=<Kepler|TESS>`

Resolves a name or ID. Lookup order: demo cache, then local light-curve cache, then MAST.

```bash
curl "localhost:8000/targets/search?q=Kepler-10"
```
```json
[{"query": "Kepler-10", "target_id": "KIC 11904151", "mission": "Kepler", "n_products": 15,
  "authors": ["Kepler"], "source": "mast"}]
```

## `POST /analyze` → `202 Accepted`

```bash
curl -X POST localhost:8000/analyze -H 'content-type: application/json' \
     -d '{"target_id": "Kepler-10", "mission": "Kepler", "detrend_method": "biweight", "search_method": "bls"}'
```

| Field | Type | Default | Notes |
|---|---|---|---|
| `target_id` | string | required | `Kepler-10`, `KIC 11904151`, `TIC 261136679`, a demo slug |
| `mission` | `Kepler` \| `TESS` | auto | |
| `detrend_method` | `biweight` \| `savgol` | `biweight` | |
| `search_method` | `bls` | `bls` | TLS is a stretch goal |
| `max_candidates` | 1-5 | 3 | iterative search depth |
| `max_segments` | 1-40 | all | limit quarters/sectors downloaded (faster) |
| `force` | bool | false | ignore result cache and demo cache |

```json
{"job_id": "5e72d8af78ac", "state": "queued", "status_url": "/jobs/5e72d8af78ac"}
```

If the same request was answered before (result cache, keyed by request + model checkpoint +
pipeline version) or the target is a demo star, the job is already `done` in this response.

## `GET /jobs/{job_id}`

Poll every ~1 s. `state` goes `queued` → `running` → `done` | `failed`; `stage` and `progress`
(0-1) describe the current step (`fetching light curve`, `searching for transits (BLS)`, ...).

```json
{
  "job_id": "5e72d8af78ac", "kind": "analyze", "state": "done", "stage": "done", "progress": 1.0,
  "created_at_utc": "2026-11-21T10:00:00+00:00", "finished_at_utc": "2026-11-21T10:00:14+00:00",
  "error": null,
  "result": {
    "query": "Kepler-10", "target_id": "KIC 11904151", "status": "ok",
    "warnings": [],
    "lightcurve": {"mission": "Kepler", "author": "Kepler", "exptime_s": 1800, "n_points": 28000,
                   "baseline_days": 560.2, "segments": [1, 2, 3, 4], "detrend_method": "biweight", "...": "..."},
    "plots": {"raw": {"x": [], "y": []}, "trend": {"x": [], "y": []}, "detrended": {"x": [], "y": []}},
    "stellar": {"radius": 1.06, "mass": 0.91, "teff": 5700, "tmag": 10.4},
    "model": {"kind": "cnn", "checkpoint": "models/final/cnn/best.pt"},
    "timings_s": {"fetch": 4.1, "search": 7.9, "classify": 0.3, "total": 13.2},
    "source": "computed", "cached": false,
    "candidates": [{
      "rank": 1, "period": 0.83749, "t0": 131.575, "t0_bjd": 2454964.575,
      "duration": 0.074, "duration_hours": 1.78, "depth": 0.000152, "depth_ppm": 152, "snr": 38.2,
      "sde": 21.0, "n_transits": 610, "significant": true,
      "probabilities": {"pc": 0.93, "eb": 0.02, "fp": 0.05},
      "predicted_class": "PC", "predicted_label": "Planet Candidate", "confidence": 0.93,
      "vetting_flags": [
        {"name": "odd_even", "passed": true, "value": 0.6, "message": "odd and even transits have consistent depths"},
        {"name": "secondary_eclipse", "passed": true, "value": 0.4, "message": "no significant secondary eclipse at phase 0.5"}
      ],
      "physical": {"planet_radius_rearth": 1.45, "planet_radius_rjup": 0.13, "fit_depth_ppm": 160, "...": "..."},
      "plot_data": {
        "periodogram": {"x": [], "y": []}, "periodogram_peak": 0.83749,
        "folded": {"points": {"x": [], "y": []}, "binned": {"x": [], "y": []}, "model": {"x": [], "y": []}},
        "global_view": {"x": [], "y": []}, "local_view": {"x": [], "y": []}
      },
      "saliency": {"method": "integrated_gradients", "target_class": "PC",
                   "global_view": [], "local_view": [], "features": {"log_depth": 0.07}},
      "explanation": {"technical": ["Periodic dip every 20.1 hours, ..."], "eli12": "Every 20.1 hours, this star gets ..."}
    }]
  }
}
```

The numbers above illustrate the shape; they are not a recorded Kepler-10 result.

Units: periods, durations and `t0` are in days; `t0` is in the mission time system (BKJD for Kepler,
BTJD for TESS) and `t0_bjd` in BJD_TDB. Folded-plot x values and the local view are in hours from
mid-transit; the global view x is in days. Saliency arrays are 0-1 magnitudes per view bin.

## `GET /demo/targets`

The pre-computed demo stars (`make warm-cache`), usable offline.

```json
[{"slug": "kepler-10", "title": "Kepler-10", "query": "Kepler-10", "target_id": "KIC 11904151",
  "mission": "Kepler", "category": "small_multi", "blurb": "...", "predicted_class": "PC",
  "confidence": 0.93, "cached_at_utc": "..."}]
```

## `GET /demo/{slug}`

The full `AnalysisResult` of one demo star (same shape as `result` above).

## `POST /hunt` → `202 Accepted` (stretch goal: hunt mode)

```bash
curl -X POST localhost:8000/hunt -H 'content-type: application/json' \
     -d '{"target_ids": ["TIC 261136679", "Kepler-10", "KIC 5780885"], "max_segments": 4}'
```

Poll `/jobs/{job_id}`; the finished job has `hunt_result.ranked`: significant signals first,
sorted by planet probability. Per-target failures are kept as rows with `status` = error code.

```json
{"ranked": [{"target_id": "KIC 11904151", "query": "Kepler-10", "status": "ok", "best_period": 0.8375,
             "best_depth_ppm": 152, "best_snr": 38.2, "planet_probability": 0.93, "predicted_class": "PC", "error": null}]}
```

## Python client example

```python
import time, httpx

api = httpx.Client(base_url="http://localhost:8000", timeout=30)
job = api.post("/analyze", json={"target_id": "Kepler-10"}).json()
while (status := api.get(job["status_url"]).json())["state"] not in ("done", "failed"):
    print(f"{status['progress']:.0%} {status['stage']}")
    time.sleep(1)
best = status["result"]["candidates"][0]
print(best["predicted_label"], best["confidence"], best["period"])
```
