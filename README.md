# GTA VI launch-day Twitch viewers — hour-by-hour forecast

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/tuyullima311-cloud/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/tuyullima311-cloud/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Entry for [Forecast GTA VI's launch-day Twitch viewers hour by hour](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) on make.fun.

## What it does

`main.py` (Python 3.13, standard library only, no network) reads one case
(`/input/<id>.in`) and writes `/output/<id>.out`: for each of hours 2–24
of the release day (UTC), five quantiles (5%, 25%, 50%, 75%, 95%) of the
Twitch concurrent viewers in the Grand Theft Auto VI category.

## Method

Launch-day audiences share a shape in logs — a night dip in hours 2–4, a
jump when the US east coast unlocks, a climb to the US prime-time peak in
hours 14–20 — but the size of the swing depends on how big the first hour
already is. So for each forecast hour the program fits, across the case's
comparable past launches, an OLS line in logs,

    log(viewers_h) = a_h + b_h · log(print_1),

and predicts from the case's own first-hour print. Two refinements:

- comparables whose first hour is nearly empty (< 500 viewers) are dropped
  from the fits — their ratios explode and would drag the slope;
- the fitted median is blended 80/20 with the pooled median log-ratio
  shape, which corrects the regression's tendency to shrink the evening
  peak too much for big launches.

The spread is the fit's residual standard deviation in log space, shrunk
by 0.55 (the raw residuals are inflated by near-empty first hours).
Deterministic: pure arithmetic, no randomness, no clock.

## Local validation

Practice answers (SullyGnome hourly averages for the four practice
launches, fetched once from the public chart API named in the kit's
`input/NOTICE.txt`, hour-1 values verified to match the case files
exactly):

| | pinball loss | typical miss | 90% coverage |
|---|---|---|---|
| this entry | **0.176** | 32.0% | 0.99 |
| house baseline (starter) | 0.387 | 82.7% | 0.78 |

Per case: p00 0.171, p01 0.186, p02 0.164, p03 0.181.

Leave-one-out over the 20 comparables with a first hour above 50,000
viewers (the scale GTA VI will be at): loss 0.232 vs 0.288 for the
unrefined regression. Simulated-final check (each practice case predicted
with the other three practice launches added to the training set, as the
final case holds them): loss 0.169.

Runtime: ~0.05 s per case, two runs byte-identical, output well under the
65,536-byte cap.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
