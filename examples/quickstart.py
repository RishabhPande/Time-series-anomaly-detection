"""Quickstart: generate a series, train the Transformer detector, compare.

Trains a from-scratch Transformer autoencoder on the clean head of a
synthetic series, then scores the whole series (including planted
anomaly episodes) and compares against a moving z-score baseline.
Prints a metric table plus an ASCII plot, and saves a PPM chart of
scores vs. true labels to examples/scores.ppm.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ts_anomaly import (
    AnomalyDetector,
    TimeSeriesGenerator,
    TransformerAutoencoder,
    moving_zscore,
    point_adjusted_f1,
    precision_recall_f1,
    roc_auc,
    scores_to_labels,
)
from ts_anomaly.train import train_autoencoder


def ascii_plot(scores, labels, threshold, width=70, height=12):
    n = len(scores)
    lo, hi = min(scores), max(scores)
    span = hi - lo or 1.0
    cols = []
    for c in range(width):
        seg = scores[c * n // width : (c + 1) * n // width + 1]
        v = max(seg) if seg else lo
        cols.append(int((v - lo) / span * (height - 1)))
    thr_row = int((threshold - lo) / span * (height - 1))
    print("\nanomaly score (top = high), '=' = threshold, 'x' = true anomaly")
    for r in range(height - 1, -1, -1):
        line = "".join(
            "=" if c == thr_row and r == thr_row else ("*" if c >= r else " ")
            for c in cols
        )
        print("  " + line)
    marks = "".join(
        "x"
        if any(labels[c * n // width : (c + 1) * n // width + 1])
        else " "
        for c in range(width)
    )
    print("  " + marks + "  <- true labels")


def save_ppm(scores, labels, threshold, path, width=700, height=240):
    """P3 PPM: score curve on top, label strip at the bottom."""
    n = len(scores)
    lo, hi = min(scores), max(scores)
    span = hi - lo or 1.0
    top = height - 40
    thr_y = top - int((threshold - lo) / span * (top - 1))
    px = [[(255, 255, 255) for _ in range(width)] for _ in range(height)]
    for c in range(width):
        seg = scores[c * n // width : (c + 1) * n // width + 1]
        v = max(seg) if seg else lo
        y = top - int((v - lo) / span * (top - 1))
        for yy in range(y, top):
            px[yy][c] = (30, 30, 30)
        lab = any(labels[c * n // width : (c + 1) * n // width + 1])
        for yy in range(top + 4, height):
            px[yy][c] = (200, 30, 30) if lab else (235, 235, 235)
    for c in range(width):
        if 0 <= thr_y < height:
            px[thr_y][c] = (30, 120, 200)
    with open(path, "w") as f:
        f.write(f"P3\n{width} {height}\n255\n")
        for row in px:
            f.write(" ".join(f"{r} {g} {b}" for r, g, b in row) + "\n")


def main():
    t0 = time.time()
    # Stationary series (no trend drift): train and test share one
    # distribution, the standard unsupervised anomaly-detection setup.
    gen = TimeSeriesGenerator(seed=7, trend_slope=0.0)
    series, labels = gen.generate(n=1500, n_episodes=10, anomaly_start=0.6)
    n_anom = sum(labels)
    print(f"generated {len(series)} points, {n_anom} labeled anomaly points "
          f"across 10 episodes")

    # Train on the clean head only: the model learns "normal".
    train_series = series[:720]
    ae = TransformerAutoencoder(window=24, d_model=12, d_ff=24, seed=42)
    history = train_autoencoder(
        ae, train_series, epochs=20, batch_size=64, lr=0.1,
        patience=5, stride=4, seed=0,
    )
    print(f"trained {history['epochs_run']} epochs "
          f"(train MSE {history['train_loss'][0]:.3f} -> "
          f"{history['train_loss'][-1]:.3f}, "
          f"best val {history['best_val_loss']:.3f})")

    detector = AnomalyDetector(ae)
    threshold = detector.fit_threshold(train_series, quantile=0.99, smooth=5)
    t_scores = detector.score(series, smooth=5)
    t_pred = scores_to_labels(t_scores, threshold)

    z_scores = moving_zscore(series, window=48)
    z_sorted = sorted(z_scores[:900])
    z_threshold = z_sorted[int(0.99 * len(z_sorted))]
    z_pred = scores_to_labels(z_scores, z_threshold)

    def report(name, scores, pred):
        prf = precision_recall_f1(labels, pred)
        pa = point_adjusted_f1(labels, pred)
        auc = roc_auc(scores, labels)
        print(f"{name:14s} P={prf['precision']:.3f} R={prf['recall']:.3f} "
              f"F1={prf['f1']:.3f} | point-adj F1={pa['f1']:.3f} "
              f"| AUC={auc:.3f}")

    print("\npoint-wise metrics (threshold = 99th pct of clean-train scores):")
    report("transformer", t_scores, t_pred)
    report("zscore-baseline", z_scores, z_pred)

    ascii_plot(t_scores, labels, threshold)
    out = os.path.join(os.path.dirname(__file__), "scores.ppm")
    save_ppm(t_scores, labels, threshold, out)
    print(f"\nsaved PPM chart to {out}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
