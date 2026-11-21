# The science, explained simply

## The transit method

When a planet passes between us and its star, it blocks a small fraction of the starlight. The
star's brightness, plotted over time (a **light curve**), shows a small dip that repeats every
orbit.

* **Depth** ≈ (planet radius / star radius)². Jupiter in front of the Sun: ~1 % (10,000 ppm).
  Earth in front of the Sun: ~0.008 % (84 ppm).
* **Period** = time between dips = the planet's orbital period (Kepler's third law then gives the
  orbit size).
* **Duration** depends on the orbit size, the star's size and the impact parameter (how
  centrally the planet crosses). A central transit across a Sun-like star lasts about
  13 h × (P / 1 yr)^(1/3).

Kepler (2009–2013) stared at about 150,000 stars for four years. TESS (2018–) covers most of the
sky in 27-day sectors, focusing on bright, nearby stars.

## From raw photons to a clean light curve

1. **PDCSAP flux.** The mission pipelines already remove many spacecraft systematics. We
   normalize each Kepler quarter or TESS sector by its own median.
2. **Cleaning.** We drop invalid points and clip *upward* outliers (flares, cosmic rays). We
   never clip downward, because transits are downward outliers.
3. **Detrending.** Stars vary: spots rotate, they pulsate, and there are slow instrumental drifts.
   A robust running "biweight" filter (`wotan`) follows these slow trends while ignoring short
   dips, as long as its window is at least 3× the transit duration. Dividing by the trend leaves
   a flat line at 1 with the transits still in it.

## Finding the period: Box Least Squares (BLS)

BLS (Kovács, Zucker & Mazeh 2002) asks, for every trial period and duration: *if I fold the
light curve at this period, is there a box-shaped dip?* For each trial it slides a box across the
folded curve and measures how much better a "dip here" model fits than a flat line. The best
period is the tallest spike in the **periodogram**.

* Aliases appear at P/2 and 2P, which is why the UI marks them.
* The **SNR** is the depth divided by its uncertainty. We call a signal significant at
  SNR ≥ 7.1 (Kepler's classic threshold) with at least two observed transits.
* **Iterative search:** after finding a signal, we mask its transits and search again. This is
  how multi-planet systems such as Kepler-10 b and c show up.

## Not every dip is a planet

| Impostor | What it looks like | How we catch it |
|---|---|---|
| **Eclipsing binary (EB)** | Two stars orbiting each other. Dips are often deep (> 2–3 %) and V-shaped. | Secondary eclipse at phase 0.5, V shape, an implied "planet" radius > 2 R_Jup |
| **EB at half the period** | Two near-equal eclipses per orbit. BLS locks onto P/2, so consecutive "transits" alternate between primary and secondary. | **Odd/even test:** odd- and even-numbered dips have different depths |
| **Background EB** | A faint EB blended with the target star. Its dip is diluted, so it looks shallow and planet-like. | Centroid shifts (not available here; part of the label noise) |
| **Systematics** | Spacecraft momentum dumps, thermal settling, detector artifacts, stellar variability that survived detrending. | Non-transit shape, inconsistent depths, dips at other phases, low SNR |

These are exactly the checks a human vetter runs on a Kepler "Data Validation" report. The app
runs them automatically and shows each one as a green or red badge.

## What the neural network sees

Following **AstroNet** (Shallue & Vanderburg 2018), each signal becomes two pictures:

* **Global view:** the whole orbit folded and binned into 201 points. It reveals secondary
  eclipses and orbit-wide structure.
* **Local view:** a zoom of ±2 transit durations binned into 61 points. It shows the transit
  shape: a flat-bottomed U for a planet, a V for a grazing EB.

A 1D convolutional network reads both views (a deeper branch for the global view, a shallower
one for the local view), adds six scalar measurements (period, duration, depth, SNR, odd/even
ratio, secondary depth), and outputs three probabilities: planet candidate, eclipsing binary,
other false positive.

**Integrated Gradients** then traces the prediction back to the input bins. A convincing planet
verdict should come from the transit bins and the flat region where a secondary eclipse would be.

## Why "candidate", not "planet"

A light curve alone can't rule out every impostor; background EBs blended in the same pixels are
the classic case. Confirming a planet needs follow-up: radial-velocity mass measurements,
high-resolution imaging, multi-color photometry, or statistical validation. This tool ranks and
explains signals so humans spend that expensive follow-up time where it matters most.

## References

* Kovács, Zucker & Mazeh (2002), *A box-fitting algorithm in the search for periodic transits*, A&A 391, 369.
* Shallue & Vanderburg (2018), *Identifying Exoplanets with Deep Learning*, AJ 155, 94.
* Hippke et al. (2019), *Wōtan: Comprehensive time-series detrending in Python*, AJ 158, 143.
* Seager & Mallén-Ornelas (2003), *A unique solution of planet and star parameters from an extrasolar planet transit light curve*, ApJ 585, 1038.
* Sundararajan, Taly & Yan (2017), *Axiomatic Attribution for Deep Networks* (Integrated Gradients), ICML.
* Thompson et al. (2018), *Planetary Candidates Observed by Kepler. VIII. DR25*, ApJS 235, 38.
