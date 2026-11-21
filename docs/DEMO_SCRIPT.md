# Demo script: 3-minute pitch + click-by-click demo

## Pre-demo checklist (T−30 min)

- [ ] `make demo` running; http://localhost:8501 open, browser zoom 110%, sidebar collapsed.
- [ ] Sidebar badge says **API online · model: cnn** (or gbm).
- [ ] All 6 demo buttons visible. Click each once so every result is warm.
- [ ] `.env` has `TH_OFFLINE=true` if the Wi-Fi is unreliable (demo stars still work).
- [ ] `models/results.md` open in a second tab (for the "results" beat and judge questions).
- [ ] Backup: a screen recording of the full flow in case the laptop fails (see *If things break*).
- [ ] Phone timer set to 3:00.

## The pitch (3:00)

> Timings are cumulative. Say the bold lines; everything else is what's on screen.

**0:00 – 0:25 · Hook**

**"NASA's TESS satellite has flagged over 8,000 stars as possibly having planets, and almost
5,000 of them are still waiting for a verdict. One machine-learning survey this year added
another 10,000 candidates from a single year of data. There aren't enough astronomers to look at
them all."**

*(On screen: landing page.)*

**0:25 – 0:45 · Problem**

**"Most of those signals aren't planets. They're pairs of stars eclipsing each other, or glitches
in the telescope. An expert checks each one by hand. We built a tool that does the first pass in
seconds and shows its reasoning, so experts only spend time on the good ones."**

**0:45 – 2:05 · Live demo**

1. *(Click **🪐 Kepler-7**.)* **"This is real NASA data: Kepler watching one star for over a
   year."** Point at ① the raw light curve. **"The orange line is the star's own variability, which
   we remove."**
2. Scroll to ② **"Now it's flat, and the orange dots are dips."**
3. ③ **"We test thousands of possible orbital periods. This spike is the winner: 4.9 days."**
4. ④ **"Stack every dip on top of each other, and there's the transit, with a flat bottom. A
   planet."**
5. ⑤ **"Our neural network says planet candidate, and every vetting check is green: same depth
   on odd and even orbits, no second eclipse, the right shape."**
6. ⑥ **"And this is why: the model is looking at the transit itself. It's not a black box."**
7. *(Click **⭐⭐ Known eclipsing binary**.)* **"Same pipeline, different star."** Point at the red
   badges. **"V-shaped dip, a second eclipse halfway around the orbit, and it would be twice the
   size of Jupiter. Two stars, not a planet. Same conclusion a human vetter would reach."**
8. *(Toggle **🧒 Explain like I'm 12**.)* **"And for students, every step explains itself in
   plain language."** Read one sentence of the verdict.

**2:05 – 2:30 · Results**

*(Switch to the `models/results.md` tab or the README results table.)*

**"We trained on [N] real Kepler signals from [M] stars, split by star so the model is always
tested on stars it has never seen. The CNN reaches [X] macro PR-AUC, against [Y] for a gradient
boosting baseline, and at 95% precision it still keeps [Z]% of real planet candidates. It runs in
under 2 seconds per star on a laptop CPU."**

> Fill the brackets from `models/results.md` after the final training run. If the GBM wins, say
> so: "the simpler model won on our data size, and our tool serves whichever is better."

**2:30 – 2:50 · Impact**

**"Point it at any TESS or Kepler star and you get a ranked, explained verdict. Hunt mode scans a
list of stars and ranks them by planet probability, so follow-up telescope time goes to the best
candidates first."**

**2:50 – 3:00 · Next steps + close**

**"Next: more TESS training data, and centroid checks to catch blended binaries. We
don't confirm planets. We decide which ones are worth a telescope's time."**

## Click-by-click reference

| Step | Action | What to point at |
|---|---|---|
| 1 | Click **Kepler-7** | Header: KIC ID, Kepler, data span |
| 2 | Scroll to ① ② | Trend line, then highlighted dips |
| 3 | ③ periodogram | Orange peak + label; dotted P/2 and 2P |
| 4 | ④ folded | Binned blue points on the orange model |
| 5 | ⑤ verdict | Class, confidence bars, green badges, "size if a planet" |
| 6 | ⑥ why | Bright, large dots sit on the transit |
| 7 | Click **Known eclipsing binary** | Red badges: shape, secondary, radius |
| 8 | Toggle ELI12 | Verdict paragraph changes |
| (spare) | Click **Pi Mensae (TESS)** | Shallow TESS super-Earth (π Men c, 6.27 d), same pipeline on TESS data |
| (spare) | Click **Kepler-10** | Two signals, b (0.84 d) and c (45 d): iterative search |

## Five likely judge questions

**1. "How do you know it isn't just memorizing stars?"**
Splits are by star: every signal from a given star lives in exactly one of train, validation or
test, and the build fails if any star leaks. Synthetic augmentation inherits the host star's split
and is excluded from test metrics. All reported numbers are on stars the model never saw.

**2. "Your labels come from humans and a pipeline. Aren't they noisy?"**
Yes, and the data card says so. Some "candidates" are undiscovered false positives, and some
eclipsing binaries hide behind other stars. That noise caps how accurate any model can get. It's
also why we show vetting checks and saliency next to every prediction instead of a bare score,
and why the output is "candidate for follow-up", never "planet".

**3. "Kepler and TESS are different telescopes. How do you handle both?"**
We train on both. TESS has no official "eclipsing binary" label, so we cross-match TESS false
positives against the published TESS eclipsing-binary catalog to get all three classes. The model
also gets a mission flag, so it can learn each telescope's noise. We report test results
separately for Kepler and TESS stars, and TESS is lower, because its labels are noisier and its
light curves are shorter.

**4. "Why a CNN? Wouldn't a simpler model do?"**
We built the simpler model too: a LightGBM baseline on 31 hand-crafted features, shown side by
side in the results table. The CNN learns shape features the hand-made ones miss, such as subtle
V-shapes and partial secondaries. The app automatically serves whichever model scores better on
validation. Both explain themselves: Integrated Gradients for the CNN, SHAP for the trees.

**5. "What did you build versus what came from libraries?"**
We used lightkurve for downloads, wotan for detrending, PyTorch and LightGBM for the models. We
built:
* the pipeline end to end, including a parallel numba BLS that's ~6× faster than astropy on
  a 2-core machine and scales with cores, on multi-year Kepler data, checked against astropy in tests;
* the three-class label mapping and the resumable dataset builder with star-level splits;
* the vetting suite, the train/serve-consistent feature path, and the trapezoid refinement that
  fixes BLS duration bias;
* the API with background jobs and caching, the explainable UI, and 80+ automated tests.

## If things break on stage

| Problem | Fix (in order) |
|---|---|
| Wi-Fi dead | Nothing to do: demo buttons read `data/demo_cache`. Avoid typing new stars. |
| API crashed | UI switches to in-process mode automatically (yellow sidebar badge). Keep clicking demo stars. |
| A new star takes too long | Say "this one is downloading four years of data, so let's use the cached one" and click Kepler-10. |
| UI crashed | `make ui` (5 s). Meanwhile, talk over the results table. |
| Laptop dead | Play the backup screen recording from a phone or a teammate's laptop. |
