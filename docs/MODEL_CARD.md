# Model card

## Intended use

* **Use it for:** ranking and explaining periodic transit-like signals in Kepler and TESS light
  curves, so people can prioritize follow-up. It's also a teaching tool for how the transit
  method and vetting work.
* **Don't use it for:** confirming planets, occurrence-rate statistics (the probabilities are not
  calibrated to the natural population), or single-transit events (the search needs ≥ 2 transits).
* **Users:** hackathon judges, students, citizen scientists, astronomers triaging candidates.

## Models

| | Dual-branch CNN (primary) | LightGBM (baseline) |
|---|---|---|
| Inputs | Global view (201 bins), local view (61 bins), 7 standardized scalars (incl. `is_tess`) | 14 scalar features + 18 view summaries |
| Architecture | Global: 4 blocks × 2 conv(k=5) + maxpool(5, s=2), 16→128 channels. Local: 2 blocks, 16→32 channels. Concatenated with scalars, then FC 256 → 128 (dropout 0.3) → 3 | Gradient-boosted trees, 31 leaves, early stopping on validation log-loss |
| Size | ~0.5 M parameters, ~2 MB checkpoint | ~2 MB |
| Training | AdamW (lr 1e-3, wd 1e-4), cosine schedule, class-weighted cross-entropy (focal loss optional), augmentation: time flip, phase shift (±3 global / ±1 local bins), Gaussian noise σ = 0.02. Early stopping on validation macro PR-AUC, patience 8 | Balanced class weights |
| Explanations | Integrated Gradients (Captum, 32 steps, zero baseline) | TreeSHAP (`pred_contrib`) |
| CPU latency | < 1 ms predict, ~15 ms with explanation | < 5 ms |

Class order: `PC` (planet candidate), `EB` (eclipsing binary), `FP` (other false positive).
`TH_MODEL_KIND=auto` serves whichever model has the higher validation macro PR-AUC.

## Evaluation

`make train` writes every number below to `models/metrics.json` and `models/results.md`.
**Paste the test table from `models/results.md` here after the final training run.** The test
split contains real Kepler signals from stars never seen in training; synthetic examples are
excluded.

| Model | Macro PR-AUC | Macro F1 | PC recall @ 95% precision | PC F1 | EB F1 | FP F1 | ECE (PC) |
|---|---|---|---|---|---|---|---|
| CNN | _fill in_ | | | | | | |
| LightGBM | _fill in_ | | | | | | |

Also reported: the same table for **Kepler-only and TESS-only test stars** (`test_kepler`,
`test_tess`), per-class precision and recall, confusion matrices, the reliability curve for
P(planet), and CPU latency.

**What "PC recall @ 95% precision" means:** pick the probability threshold at which 95% of the
signals called "planet" really are planet candidates. What fraction of all planet candidates
still clear that bar? This is the operating point for a follow-up queue that can't afford to
waste telescope time.

For context, AstroNet (Shallue & Vanderburg 2018) reported about 0.98 AUC on a *binary*
planet-vs-not task using the full Kepler DR24 TCE set. Our task has three classes, is trained on
a class-balanced subset of hundreds of stars, and runs on a laptop CPU, so expect lower numbers.
Compare against the LightGBM baseline instead.

## Limitations

* **Label noise caps accuracy.** KOI CANDIDATEs include undetected false positives, and
  background EBs are often indistinguishable from planets without centroid or follow-up data.
  See the data card.
* **Two missions, unequal quality.** Trained on Kepler and TESS together, with a mission flag.
  TESS labels are noisier (unvetted TOI candidates; EB labels from a catalog cross-match) and
  TESS views have fewer transits. Check `test_tess` vs `test_kepler` in `models/results.md`.
* **Small training set.** Hackathon-sized (hundreds to a few thousand signals). Rare morphologies
  (heartbeat stars, eccentric EBs, single deep events) are under-represented.
* **Calibration.** Trained on a class-balanced sample, so P(planet) is not the probability that a
  random signal on the sky is a planet. Check the ECE and reliability curve.
* **Upstream errors propagate.** A wrong BLS period (alias, harmonic) produces wrong views. The
  periodogram and odd/even badge make this visible but don't fix it.
* **Saliency is a diagnostic, not proof.** Integrated Gradients shows what drove the score, not
  whether the score is right.

## Ethical considerations

Low risk: public astronomical data, no personal data. The main risk is over-claiming. The UI
always says "candidate", shows confidence and vetting flags next to the verdict, and states that
the tool doesn't confirm planets.
