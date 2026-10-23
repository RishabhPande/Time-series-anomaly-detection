"""Reconstruction-based anomaly detector over a Transformer encoder.

The autoencoder is trained on *normal* windows only. At inference time,
windows that the model cannot reconstruct well get high anomaly scores.
Because the score is the reconstruction error of the whole window, a
collective anomaly (a segment whose shape is wrong even though each
value looks fine) scores high, while a fixed per-value threshold would
miss it.
"""

from __future__ import annotations

from .attention import TransformerEncoder


class TransformerAutoencoder:
    """Thin wrapper: reconstruct a window, report mean squared error.

    Inputs are standardized with the mean/std of the normal training
    data (fit via ``fit_normalizer``) so the encoder always works in a
    well-conditioned space; outputs are mapped back before scoring.
    """

    def __init__(
        self,
        window: int,
        d_model: int = 16,
        d_ff: int = 32,
        seed: int = 42,
    ) -> None:
        self.encoder = TransformerEncoder(window, d_model, d_ff, seed=seed)
        self.window = window
        self.mean = 0.0
        self.std = 1.0

    def fit_normalizer(self, train_series: list[float]) -> None:
        n = len(train_series)
        mean = sum(train_series) / n
        var = sum((x - mean) ** 2 for x in train_series) / n
        self.mean = mean
        self.std = max(var ** 0.5, 1e-6)

    def standardize(self, window: list[float]) -> list[float]:
        return [(x - self.mean) / self.std for x in window]

    def reconstruct(self, window: list[float]) -> list[float]:
        r = self.encoder.forward(self.standardize(window))
        return [v * self.std + self.mean for v in r]

    def window_error(self, window: list[float]) -> float:
        recon = self.reconstruct(window)
        return sum((r - x) ** 2 for r, x in zip(recon, window)) / len(window)


class AnomalyDetector:
    """Per-timestep anomaly scores from window reconstruction error.

    Scores each timestep by averaging the squared reconstruction error
    over every sliding window that covers it. ``fit_threshold`` picks a
    flag threshold as a high quantile of per-timestep errors on known
    normal data, the standard unsupervised protocol.
    """

    def __init__(self, autoencoder: TransformerAutoencoder) -> None:
        self.ae = autoencoder
        self.window = autoencoder.window

    def score(self, series: list[float], smooth: int = 0) -> list[float]:
        n = len(series)
        w = self.window
        acc = [0.0] * n
        cnt = [0] * n
        for start in range(0, n - w + 1):
            win = series[start : start + w]
            recon = self.ae.reconstruct(win)
            for i in range(w):
                err = (recon[i] - win[i]) ** 2
                acc[start + i] += err
                cnt[start + i] += 1
        scores = [a / c if c else 0.0 for a, c in zip(acc, cnt)]
        if smooth > 1:
            scores = _moving_average(scores, smooth)
        return scores

    def fit_threshold(
        self, normal_series: list[float], quantile: float = 0.99, smooth: int = 0
    ) -> float:
        errors = sorted(self.score(normal_series, smooth=smooth))
        idx = min(len(errors) - 1, int(quantile * len(errors)))
        return errors[idx]

    def predict(
        self, series: list[float], threshold: float, smooth: int = 0
    ) -> list[int]:
        return [1 if s >= threshold else 0 for s in self.score(series, smooth)]


def _moving_average(xs: list[float], k: int) -> list[float]:
    """Centered moving average so smoothed peaks do not lag the labels."""
    half = k // 2
    return [
        sum(xs[max(0, i - half) : i + half + 1])
        / len(xs[max(0, i - half) : i + half + 1])
        for i in range(len(xs))
    ]
