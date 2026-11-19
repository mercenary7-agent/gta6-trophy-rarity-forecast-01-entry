"""gta6-trophy-rarity-forecast-01 entry: per-trophy logit-trend forecaster (Python 3.13, stdlib only).

Reads the case in /input, forecasts each trophy's Exophase earn rate (percent) at the
case's target as five quantiles (0.05/0.25/0.50/0.75/0.95), writes /output/<id>.out.

Model (v1): each trophy's rate is transformed to the logit scale (clamped 0.01..99.99).
The center is the last observed logit plus a shrunk linear trend fitted over the most
recent reads (slope shrunk toward 0 by SHRINK, since launch-night trends are noisy),
extrapolated to the target. Quantiles are normal quantiles around the center with
spread S0 * sqrt(horizon_days) in logit units, mapped back through the logistic.
Fully deterministic: no randomness, no network, stdlib only.
"""

import json
import math
import os
import sys
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Spread of the rate's move in logit units per square-root day.
S0 = 0.5
# Trend shrinkage: fraction of the fitted logit slope kept.
SHRINK = 0.55
# How many of the most recent reads the trend is fitted on.
TREND_READS = 16
# Clamp bounds for rates (percent).
RLO, RHI = 0.01, 99.99


def parse_ts(s):
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))


def logit(p):
    p = min(max(p, RLO), RHI) / 100.0
    return math.log(p / (1.0 - p))


def logistic(x):
    return 100.0 / (1.0 + math.exp(-x))


def fit_slope(points):
    """Weighted least-squares slope of logit rate vs days, over the last TREND_READS points."""
    pts = points[-TREND_READS:]
    n = len(pts)
    if n < 2:
        return 0.0
    # recency weights: linear ramp, newest weight 2, oldest 1
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ws = [1.0 + (i / max(n - 1, 1)) for i in range(n)]
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    num = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    den = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if den <= 1e-12:
        return 0.0
    return num / den


def forecast_case(case):
    at = parse_ts(case["at"])
    target = parse_ts(case["target"])
    d_days = max((target - at).total_seconds() / 86400.0, 0.0)
    hist = case.get("history") or []
    trophies = case["trophies"]

    # index reads by time; rates align with trophies order
    reads = []
    for rd in hist:
        t = parse_ts(rd["at"])
        reads.append((t, rd.get("rates") or []))
    reads.sort(key=lambda r: r[0])

    out = []
    for ti, tr in enumerate(trophies):
        pts = []
        for t, rates in reads:
            if ti < len(rates) and rates[ti] is not None:
                pts.append(((t - at).total_seconds() / 86400.0, logit(rates[ti])))
        if not pts:
            center = 0.0
        else:
            last_x = pts[-1][0]
            last_y = pts[-1][1]
            slope = SHRINK * fit_slope(pts)
            center = last_y + slope * (d_days - last_x)
        sigma = S0 * math.sqrt(d_days)
        qs = []
        for z in Z:
            v = logistic(center + z * sigma)
            qs.append(min(100.0, max(0.0, round(v, 4))))
        # enforce non-decreasing
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        out.append({"id": tr["id"], "q": qs})
    return {"trophies": out}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(f.read_text())
        res = forecast_case(case)
        (OUTPUT_DIR / (case["id"] + ".out")).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
