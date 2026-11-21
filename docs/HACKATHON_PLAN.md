# Hackathon plan: 18 hours, 4 people

The code is written before the event. The 18 hours go to **real data, training, tuning, the
demo, and the pitch**.

**Roles:** **A** (ML), **B** (data), **C** (backend/integration), **D** (frontend, docs, pitch).

## Night before (allowed: pre-downloading public data)

```bash
make setup && make test && make test-network   # all green before anything else
make data-download LIMIT=800 TESS_LIMIT=400 WORKERS=8 PURGE=1  # Kepler + TESS; overnight; resumable
```

Check `make data-status` after 15 minutes. If a star takes longer than ~20 s, either lower `LIMIT`
or leave it running longer. Also commit `uv.lock` (created by `make setup`) so every laptop
installs identical versions.

## Hour by hour

| Hour | A (ML) | B (Data) | C (Backend) | D (Frontend/Pitch) |
|---|---|---|---|---|
| 0–1 | All: pull, `make setup`, `make test`, agree on the target class balance and the demo star list | ← | ← | ← |
| 1–2 | Read `dataset_meta.json` counts; sanity-check views of 10 random signals per class (notebook 01) | `make data-process`; read `data/interim/errors_real.txt` and fix the top failure if it's cheap | `make api`; hit every endpoint with a real star; time Kepler vs TESS | Run the UI against real data, list visual issues |
| **2** | **✅ CP1: real dataset built, counts look sane** | | | |
| 2–4 | `make train`; read `models/results.md`; confusion matrix review | `make data-tess` (optional domain-shift set); start `make data-synth` only if < 400 stars | `make export-model && make warm-cache`; check every demo star's verdict | Fix UI issues; pick final demo order |
| **4** | **✅ CP2: first real CNN vs GBM table** | | | |
| 4–8 | Tune: focal vs weighted CE, dropout, epochs; one change at a time, log every run in `results.md` | Investigate misclassified test signals (notebook 03); flag label-noise examples for the pitch | Hunt mode on 20 TOIs; speed pass if a Kepler search takes > 20 s (`bin_minutes: 60`) | Fill README results table; record the GIF |
| **8** | **✅ CP3: model frozen** (no retraining after this) | | | |
| 8–11 | Calibration and threshold for "PC recall @ 95% precision"; write MODEL_CARD numbers | DATA_CARD final counts and query dates | `make warm-cache` with the frozen model; offline test with `TH_OFFLINE=true` and Wi-Fi off | Pitch draft with real numbers; slides only if the event needs them |
| **11** | **✅ CP4: offline demo works end to end** | | | |
| 11–14 | Prepare answers for model questions | Prepare answers for data questions | Docker build (`make docker-build`) as a backup run path | Rehearse ×2 with a timer; record a backup video |
| **14** | **✅ CP5: code freeze**. After this: bug fixes only, each one tested with `make check` | | | |
| 14–16 | All: full rehearsal ×3; judge Q&A drill (DEMO_SCRIPT has 5 questions, write 5 more) | ← | ← | ← |
| 16–18 | Buffer. Presenter rests their voice; others watch the logs | ← | ← | ← |

## Fallbacks

| Trigger | Action |
|---|---|
| < 400 real stars by hour 2 | `make data-synth` (+600 synthetic examples), then `make train`. Test metrics stay real-only. |
| CNN below GBM at hour 8 | Ship it anyway with `TH_MODEL_KIND=auto` (serves GBM). Pitch it as "the baseline won at this data size". |
| MAST slow or down on the day | Everything needed is already cached; set `TH_OFFLINE=true`. |
| A demo star gets a wrong verdict | Swap it in `configs/demo_targets.yaml` and re-run `uv run python scripts/warm_demo_cache.py --only <slug>`. Or keep it and own it in the pitch: "here's a hard case, and here's what the vetting flags tell us". |
| API instability | The UI falls back to in-process automatically. |
