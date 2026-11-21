# Data card

## Sources

| Data | Provider | Access | Used for |
|---|---|---|---|
| Kepler long-cadence light curves (PDCSAP, 29.4 min) | NASA MAST | `lightkurve.search_lightcurve(..., author="Kepler", exptime=1800)` | Training, validation, test |
| TESS light curves: SPOC 2-min, then TESS-SPOC FFI, then QLP | NASA MAST | `lightkurve`, author priority set in `configs/data.yaml`; up to 4 sectors per training star | Training, validation, test, inference |
| KOI cumulative table | NASA Exoplanet Archive (TAP `cumulative`) | `data/labels.py` | Kepler labels and catalog ephemerides |
| TOI table (TFOPWG dispositions) | NASA Exoplanet Archive (TAP `toi`) | `data/labels.py` | TESS labels and ephemerides |
| TESS Eclipsing Binary catalog (Prša et al. 2022; 4,584 EBs, sectors 1–26) | VizieR `J/ApJS/258/16` | `data/labels.py` (`astroquery.vizier`) | TESS EB labels |
| TESS Input Catalog (stellar radius, mass, Teff) | MAST Catalogs | `astroquery.mast.Catalogs` | Planet-radius estimate and radius vetting flag |

The KOI cumulative table was last updated by the Kepler project on 2018-09-27; Kepler labels are
effectively frozen. Each build records the exact TAP query and the UTC query time in
`data/raw/labels/*.meta.json`, and copies them into `data/processed/dataset_meta.json`, together
with a hash of the data config.

## Label mapping

| Source disposition | Class | Notes |
|---|---|---|
| KOI `CONFIRMED`, `CANDIDATE` | **PC** (planet candidate) | Includes unconfirmed candidates. Some are false positives no one has caught yet. |
| KOI `FALSE POSITIVE` and `koi_fpflag_ss = 1` | **EB** (eclipsing binary) | "Stellar eclipse" flag. Kept as EB even when the centroid-offset flag is also set (a background EB is still an EB). |
| KOI `FALSE POSITIVE`, `koi_fpflag_ss = 0` | **FP** (other false positive) | Not transit-like (`fpflag_nt`), centroid offset, ephemeris match / contamination (`fpflag_ec`) |
| KOI `NOT DISPOSITIONED`, missing ephemeris | dropped | |
| TOI `CP`, `KP`, `PC`, `APC` | PC | |
| TOI `FP`, `FA` whose TIC is in the TESS EB catalog at a matching period (or 2× / ½×) | EB | TFOPWG has no EB disposition, so the EB catalog supplies it |
| other TOI `FP`, `FA` | FP | |
| TESS EB catalog stars that are not TOIs | EB | Added (catalog ephemeris) until EBs roughly match the PC count |
| any TESS signal with P > 13 d | dropped | Too few transits in the ≤ 4 sectors downloaded per star |

## Sampling and splits

* `make data-download LIMIT=N` picks **N stars balanced across the three classes** by each star's
  primary label (priority EB > FP > PC), seeded from `split.seed`. Every labeled signal on a
  chosen star is used.
* Splits are **by star**: 80 / 10 / 10 train / val / test, stratified by primary label and computed
  from real signals only. `finalize` refuses to write a dataset where any star appears in two
  splits.
* Synthetic examples (`make data-synth`) inherit their host star's split and are excluded from
  reported test metrics (`subset("test", real_only=True)`).
* Missions listed in `labels.train_missions` (default: Kepler and TESS) are split **per mission**,
  so validation and test always contain both. Remove TESS from that list to keep it as a held-out
  `tess` split for measuring domain shift instead.
* Each example carries an `is_tess` feature (0/1) so one model can learn mission-specific noise.

## Preprocessing (identical for training and inference)

Per-segment median normalization → NaN removal → upward 4σ clipping against a running median →
wotan biweight (window = max(1 d, 3 × longest transit on the star)) → views at the catalog
ephemeris: global (201 bins, full phase), local (61 overlapping bins, ±2 durations, bin width
0.16 durations), each normalized to median 0 and minimum −1 → 13 scalar features (see
`processing/features.py`).

## Known biases and limitations

* **Label noise.** CANDIDATE labels include undetected false positives. Some real planets carry
  the stellar-eclipse flag (hot Jupiters whose thermal secondary eclipse is detectable).
  Background EBs labeled EB can look exactly like planets in the light curve. No model trained on
  these labels can be more accurate than the labels themselves.
* **Selection bias.** Every KOI already passed Kepler's own pipeline threshold (MES ≥ 7.1).
  The model has never seen the faint, messy signals below that threshold, or random noise
  windows; BLS's significance threshold is the gate for those.
* **Class balance.** The balanced star sample doesn't match the natural class ratio, so
  probabilities are calibrated to the balanced sample, not the sky. The model card reports ECE;
  re-weight before using probabilities as population rates.
* **Kepler vs TESS differences.** Kepler gives 4-year baselines at 29.4-min cadence on mostly
  faint stars. TESS gives 27-day sectors at 2- to 30-min cadence, larger pixels (more blending)
  and different systematics, so TESS views have fewer transits and more noise. Both missions are
  in training, but TESS labels are noisier still: TOI "PC" includes many unvetted candidates, and
  EB labels depend on a catalog cross-match. Test metrics are reported per mission. If a model
  is trained without a mission, the UI warns on that mission's targets.
* **TESS EB coverage.** The EB catalog covers 2-min targets in sectors 1–26 only, so EBs among
  later or FFI-only TOIs fall back to FP.
* **Cadence.** Kepler training data is long cadence only. TESS 2-min data is binned to 30 min
  for BLS, but the views use the native cadence, so local views of short TESS transits are
  better sampled than anything seen in training.
* **Ephemeris source.** Training uses catalog ephemerides; inference uses BLS plus a
  trapezoid-fit refinement. A residual mismatch remains for low-SNR signals, where BLS durations
  are less precise.
* **Stellar parameters.** Planet radii use TIC radii, which carry ~10% or larger uncertainty
  and are wrong for unresolved binaries.

## Licenses and acknowledgements

* Kepler and TESS data are publicly available from MAST (NASA mission data, no usage
  restrictions). Please include the acknowledgement text below in any publication.
* NASA Exoplanet Archive tables: please acknowledge the archive (text below).
* Software: lightkurve (MIT), wotan (MIT), astropy (BSD-3), astroquery (BSD-3), PyTorch (BSD-3),
  LightGBM (MIT), Captum (BSD-3), FastAPI (MIT), Streamlit (Apache-2.0), Plotly (MIT).

> TESS eclipsing-binary labels: Prša, A., et al. 2022, *TESS Eclipsing Binary Stars. I.*, ApJS 258, 16.
>
> This research has made use of the NASA Exoplanet Archive, which is operated by the California
> Institute of Technology, under contract with the National Aeronautics and Space Administration
> under the Exoplanet Exploration Program.
>
> This work includes data collected by the Kepler and TESS missions, obtained from the MAST data
> archive at the Space Telescope Science Institute (STScI). Funding for the Kepler and TESS
> missions is provided by the NASA Science Mission Directorate. STScI is operated by the
> Association of Universities for Research in Astronomy, Inc., under NASA contract NAS 5-26555.
