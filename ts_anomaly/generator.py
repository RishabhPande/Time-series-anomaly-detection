"""Synthetic time-series generator with planted anomaly episodes.

The series combines daily and weekly seasonal harmonics, a slow trend,
and Gaussian noise. Anomaly episodes are injected in the tail of the
series so the head can be used as clean training data:

- point spikes: single-timestep jumps (easy, every method catches these)
- level shifts: sustained step offsets
- collective anomalies: sustained segments whose *shape* is wrong
  (e.g. suppressed seasonality) while individual values stay plausible
- contextual anomalies: plausible values at the wrong time
  (e.g. a daytime peak in the middle of the night)

All randomness goes through a seeded ``random.Random`` so datasets are
reproducible.
"""

from __future__ import annotations

import math
import random


def _gaussian(rng: random.Random) -> float:
    """One standard-normal draw via Box-Muller."""
    u1 = rng.random()
    u2 = rng.random()
    while u1 == 0.0:  # avoid log(0)
        u1 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


class TimeSeriesGenerator:
    """Seedable generator of normal series + labeled anomaly episodes."""

    KINDS = ("point", "level", "collective", "contextual")

    def __init__(
        self,
        seed: int = 42,
        daily_period: int = 24,
        weekly_period: int = 168,
        noise_std: float = 0.35,
        trend_slope: float = 0.002,
        base_level: float = 10.0,
    ) -> None:
        self.seed = seed
        self.daily_period = daily_period
        self.weekly_period = weekly_period
        self.noise_std = noise_std
        self.trend_slope = trend_slope
        self.base_level = base_level

    def _seasonal(self, t: int) -> float:
        dp = self.daily_period
        wp = self.weekly_period
        return (
            1.5 * math.sin(2.0 * math.pi * t / dp)
            + 0.6 * math.sin(4.0 * math.pi * t / dp)
            + 0.8 * math.sin(2.0 * math.pi * t / wp)
        )

    def generate(
        self,
        n: int = 1500,
        n_episodes: int = 8,
        anomaly_start: float = 0.6,
        kinds: tuple[str, ...] | list[str] | None = None,
    ) -> tuple[list[float], list[int]]:
        """Return ``(series, labels)`` with point-level 0/1 labels.

        Episodes are injected at or after ``anomaly_start * n`` so
        ``series[:int(n * anomaly_start)]`` is guaranteed clean.
        """
        rng = random.Random(self.seed)
        kinds = list(kinds) if kinds else list(self.KINDS)

        seasonal = [self._seasonal(t) for t in range(n)]
        series = [
            self.base_level
            + self.trend_slope * t
            + seasonal[t]
            + self.noise_std * _gaussian(rng)
            for t in range(n)
        ]
        labels = [0] * n

        region_start = int(n * anomaly_start)
        used: list[tuple[int, int]] = []  # occupied intervals, kept disjoint

        def _free(start: int, length: int) -> bool:
            end = start + length
            if start < region_start or end > n:
                return False
            return all(end <= s or start >= e for s, e in used)

        def _claim(start: int, length: int) -> None:
            used.append((start, start + length))

        order = [kinds[i % len(kinds)] for i in range(n_episodes)]
        rng.shuffle(order)

        for kind in order:
            placed = False
            for _ in range(200):  # rejection sampling for a free slot
                if kind == "point":
                    t = rng.randrange(region_start, n)
                    if not _free(t, 1):
                        continue
                    series[t] += rng.choice((-1.0, 1.0)) * 6.0 * self.noise_std
                    labels[t] = 1
                    _claim(t, 1)
                    placed = True
                    break
                if kind == "level":
                    length = rng.randint(20, 60)
                    t = rng.randrange(region_start, n - length)
                    if not _free(t, length):
                        continue
                    shift = rng.choice((-1.0, 1.0)) * (1.5 + rng.random())
                    for i in range(t, t + length):
                        series[i] += shift
                        labels[i] = 1
                    _claim(t, length)
                    placed = True
                    break
                if kind == "collective":
                    length = rng.randint(40, 120)
                    t = rng.randrange(region_start, n - length)
                    if not _free(t, length):
                        continue
                    # Suppress the seasonal harmonics: the segment keeps
                    # plausible values but its shape goes flat and wrong.
                    for i in range(t, t + length):
                        series[i] -= 0.9 * seasonal[i]
                        labels[i] = 1
                    _claim(t, length)
                    placed = True
                    break
                if kind == "contextual":
                    # Plant a daytime-sized peak where the daily cycle is
                    # near its trough: right value, wrong time.
                    t = rng.randrange(region_start, n - 3)
                    if not _free(t, 3):
                        continue
                    if self._seasonal(t) > -0.8:  # not night enough, retry
                        continue
                    bump = 2.2
                    for j, w in enumerate((0.5, 1.0, 0.5)):
                        series[t + j] += w * bump
                        labels[t + j] = 1
                    _claim(t, 3)
                    placed = True
                    break
            if not placed:
                raise RuntimeError(
                    f"could not place a {kind!r} episode after 200 tries; "
                    "use a longer series or fewer episodes"
                )

        return series, labels
