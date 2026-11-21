# Architecture

## System overview

```mermaid
flowchart LR
    user([User]) --> ui[Streamlit UI<br/>frontend/app.py]
    ui -- "POST /analyze, poll /jobs/{id}" --> api[FastAPI<br/>api/main.py, routes.py]
    ui -. "API down: in-process fallback" .-> pipe
    api --> jobs[JobManager<br/>thread pool]
    jobs --> pipe[pipeline.analyze]
    api <--> rcache[(Result cache<br/>diskcache)]
    pipe <--> demo[(Demo cache<br/>data/demo_cache/*.json)]
    pipe <--> lccache[(Light-curve cache<br/>data/interim/lightcurves/*.npz)]
    pipe -- "lightkurve / astroquery" --> mast[(NASA MAST<br/>Kepler, TESS, TIC)]
    pipe --> model[CNN or GBM<br/>models/final]
```

* **One code path for analysis.** The API, the UI's fallback, `scripts/hunt.py` and
  `scripts/warm_demo_cache.py` all call `transit_hunter.pipeline.analyze`.
* **One code path from signal to model input.** Training (`data/build.py`) and inference
  (`pipeline.py`) both call `processing.features.build_example`, so the views and features can't
  drift apart.
* **Three caches, cheapest first.**
  1. The demo cache holds full results, so demos work offline.
  2. The result cache is keyed by request, model checkpoint and pipeline version.
  3. The light-curve cache holds arrays, so a target is never downloaded twice.

## Inference data flow

```mermaid
flowchart TD
    q[Query: name / KIC / TIC] --> f[fetch.py<br/>resolve, download, stitch per-segment medians]
    f --> c[clean.py<br/>NaNs, sort, normalize, clip upward outliers]
    c --> d[detrend.py<br/>wotan biweight, window 1 d]
    d --> s[search.py<br/>numba BLS, iterative: find, mask, repeat]
    s --> loop{{for each candidate}}
    loop --> r[re-detrend with window >= 3 x duration<br/>trapezoid fit refines t0 and T14]
    r --> b[features.build_example]
    b --> v1[fold.py<br/>global 201 bins, local 61 bins]
    b --> v2[vetting/diagnostics.py<br/>odd/even, secondary, shape, duration, radius]
    v1 --> m[predict.py<br/>CNN or GBM, probabilities]
    v2 --> m
    m --> x[explain.py<br/>Integrated Gradients / TreeSHAP]
    x --> out[AnalysisResult JSON<br/>plots, flags, saliency, plain-language text]
```

## Training data flow

```mermaid
flowchart LR
    tap[(Exoplanet Archive TAP<br/>KOI cumulative, TOI)] --> lab[labels.py<br/>3-class mapping]
    lab --> samp[class-balanced star sample<br/>--limit N]
    samp --> dl[download stage<br/>resumable manifest, parallel]
    dl --> proc[process stage<br/>catalog ephemerides -> build_example]
    proc --> parts[(processed/parts/*.npz, *.parquet)]
    syn[synth stage<br/>injected PC / EB] --> parts
    parts --> fin[finalize<br/>star-level splits, metadata]
    fin --> ds[(dataset.npz + index.parquet)]
    ds --> tg[train GBM]
    ds --> tc[train CNN]
    tg --> ev[evaluate<br/>metrics.json, results.md]
    tc --> ev
    ev --> fm[(models/final)]
```

## Request lifecycle

```mermaid
sequenceDiagram
    participant UI
    participant API
    participant Jobs as JobManager
    participant P as pipeline
    participant MAST
    UI->>API: POST /analyze {target_id}
    alt in result cache or demo cache
        API-->>UI: 202 {job_id, state: done}
    else
        API->>Jobs: submit
        API-->>UI: 202 {job_id, state: queued}
        Jobs->>P: analyze(progress=callback)
        P->>MAST: search + download (cached after first time)
        loop every 0.6 s
            UI->>API: GET /jobs/{id}
            API-->>UI: {state: running, stage, progress}
        end
        P-->>Jobs: AnalysisResult
        Jobs->>API: store in result cache
    end
    UI->>API: GET /jobs/{id}
    API-->>UI: {state: done, result}
```

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | Typed settings (`TH_*` env vars) and validated YAML configs |
| `data/fetch.py` | Target parsing, MAST search/download, retries, light-curve cache, TIC lookup |
| `data/labels.py` | TAP queries, KOI/TOI → PC/EB/FP mapping, balanced star sampling |
| `data/build.py` | Dataset stages: labels, download, process, synth, finalize |
| `data/dataset.py` | Processed-dataset storage, star-level splits, PyTorch `Dataset` + augmentation |
| `data/synthetic.py` | Synthetic light curves and transit/EB injection (tests, fallback data) |
| `processing/clean.py` | Cleaning and normalization |
| `processing/detrend.py` | wotan biweight / Savitzky–Golay detrending |
| `processing/search.py` | Period grid, numba + astropy BLS engines, iterative search |
| `processing/fold.py` | Phase folding, global/local views |
| `processing/features.py` | Scalar features and `build_example` (the single signal → model-input function) |
| `vetting/diagnostics.py` | Odd/even, secondary eclipse, off-phase dip, shape, duration, radius |
| `models/cnn.py` | Dual-branch CNN, feature scaler |
| `models/gbm.py` | LightGBM baseline, view summaries, TreeSHAP |
| `models/train.py` | Training loops, focal loss, class weights, checkpoints |
| `models/evaluate.py` | Metrics, comparison table |
| `models/predict.py` | Common predictor interface, model selection (`auto`) |
| `models/explain.py` | Integrated Gradients (CNN), TreeSHAP (GBM) |
| `pipeline.py` | `analyze()`, `hunt()`, demo cache, trapezoid fit, plain-language explanations |
| `api/` | FastAPI app, routes, job manager, result cache, schemas |
| `frontend/` | Streamlit app, Plotly figures, backend client with in-process fallback |

## Design decisions

| Decision | Why |
|---|---|
| Train on catalog ephemerides, infer on BLS ephemerides + trapezoid refinement | Keeps BLS errors out of the labels. Refining T14 at inference matches the catalog durations used in training. Without it, BLS boxes run ~25% short and V-shaped EBs look U-shaped. |
| Own numba BLS (astropy kept as reference) | 4 years of Kepler data needs ~250k trial periods. astropy takes ~0.7 ms per period; the numba version runs in parallel on every core. A test checks the two engines agree. |
| Splits by star, from real signals only | Multi-planet systems can't leak across splits, and adding synthetic data never reshuffles the real test set. |
| Views normalized to median 0, minimum −1 | AstroNet convention. Absolute depth goes in as a scalar feature, so the views carry only shape. |
| Thread-pool jobs, not Celery/Redis | One-process deployment for a hackathon. numba already parallelizes BLS across cores. |
| Streamlit + Plotly | Fastest path to an interactive, story-driven UI. The API is UI-agnostic, so a React front end can replace it later. |
