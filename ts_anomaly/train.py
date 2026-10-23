"""Training loop for the Transformer autoencoder.

Windows are cut from the *normal* part of the series only: the detector
learns what normal looks like, and anything it cannot reconstruct is
anomalous. Mini-batch SGD with a global gradient-norm clip and early
stopping on a validation split.
"""

from __future__ import annotations

import copy
import random

from .model import TransformerAutoencoder


def make_windows(
    series: list[float], window: int, stride: int = 1
) -> list[list[float]]:
    """Cut overlapping windows of length ``window`` every ``stride`` steps."""
    return [
        series[s : s + window]
        for s in range(0, len(series) - window + 1, stride)
    ]


def train_val_split(
    windows: list[list[float]], val_frac: float = 0.2, seed: int = 0
) -> tuple[list[list[float]], list[list[float]]]:
    rng = random.Random(seed)
    idx = list(range(len(windows)))
    rng.shuffle(idx)
    cut = int(len(windows) * (1.0 - val_frac))
    tr = [windows[i] for i in idx[:cut]]
    va = [windows[i] for i in idx[cut:]]
    return tr, va or tr[:1]


def _mse(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)


def train_autoencoder(
    ae: TransformerAutoencoder,
    train_series: list[float],
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 0.03,
    patience: int = 6,
    stride: int = 2,
    seed: int = 0,
    denoise_std: float = 0.8,
) -> dict:
    """Train on normal windows; return a history dict.

    Denoising objective: each training window gets Gaussian noise added
    to its input while the target stays clean. The model must learn the
    normal temporal pattern instead of the identity map (which would
    reconstruct anomalies just as well as normal data and defeat the
    detector). Early stopping watches validation MSE and restores the
    best-seen parameters, so the returned model is the best one, not
    the last.
    """
    windows = make_windows(train_series, ae.window, stride=stride)
    train_w, val_w = train_val_split(windows, seed=seed)
    rng = random.Random(seed)
    enc = ae.encoder

    # Standardize with normal-data stats so the encoder trains in a
    # well-conditioned space; the standardized targets have mean ~0,
    # which matches the output layer's zero initialization.
    ae.fit_normalizer(train_series)

    best_params = None
    best_val = float("inf")
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        rng.shuffle(train_w)
        enc.zero_grad()
        epoch_loss, n_seen = 0.0, 0
        for i, win in enumerate(train_w):
            z = ae.standardize(win)
            # noisy input, clean target: the denoising objective
            z_in = [v + rng.gauss(0.0, denoise_std) for v in z]
            recon = enc.forward(z_in)
            loss = _mse(recon, z)
            epoch_loss += loss
            n_seen += 1
            grad = [2.0 * (r - x) / len(z) for r, x in zip(recon, z)]
            enc.backward(grad)
            if (i + 1) % batch_size == 0 or i + 1 == len(train_w):
                enc.sgd_step(lr)
                enc.zero_grad()
        train_loss = epoch_loss / n_seen
        val_loss = sum(ae.window_error(w) for w in val_w) / len(val_w)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            wait = 0
            best_params = copy.deepcopy([p.value for p in enc.parameters()])
        else:
            wait += 1
            if wait >= patience:
                break

    if best_params is not None:  # restore the best checkpoint
        for p, v in zip(enc.parameters(), best_params):
            p.value = v
    history["best_val_loss"] = best_val
    history["epochs_run"] = len(history["train_loss"])
    return history
