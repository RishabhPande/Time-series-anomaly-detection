"""Moving z-score baseline for comparison.

The classic fixed-threshold detector: each point is scored by how many
local standard deviations it sits from its trailing window mean. It
catches point spikes reliably and misses collective/contextual anomalies
whose individual values are unremarkable, which is exactly the gap the
Transformer reconstruction detector is meant to close.
"""

from __future__ import annotations

import math


def moving_zscore(series: list[float], window: int = 48) -> list[float]:
    """|x - trailing_mean| / trailing_std for each timestep."""
    scores = [0.0] * len(series)
    for i in range(len(series)):
        start = max(0, i - window)
        seg = series[start:i]
        if len(seg) < 2:
            continue
        mean = sum(seg) / len(seg)
        var = sum((x - mean) ** 2 for x in seg) / len(seg)
        scores[i] = abs(series[i] - mean) / (math.sqrt(var) + 1e-9)
    return scores
