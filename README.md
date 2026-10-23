# Time-series-anomaly-detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RishabhPande/Time-series-anomaly-detection/blob/main/examples/quickstart.py)

Transformer-based **time-series anomaly detection**, built from scratch in pure Python, zero dependencies.

A fixed threshold catches point spikes and little else. The interesting
failures in production telemetry are *collective* anomalies (a segment
whose shape is wrong even though every value looks plausible, like a
sensor whose daily cycle flattens during an outage) and *contextual*
anomalies (a plausible value at the wrong time, like peak-hour traffic
at 3 AM). `ts_anomaly` trains a small Transformer autoencoder on normal
windows only and scores each timestep by reconstruction error, so a
window the model cannot reconstruct flags itself no matter which
statistic you would have thresholded.

## Quickstart

```bash
python examples/quickstart.py
```

Generates a 1500-point synthetic series with 10 planted anomaly
episodes, trains the Transformer on the clean head, scores the full
series, compares against a moving z-score baseline, prints an ASCII
plot, and writes `examples/scores.ppm` (score curve plus true-label
strip).

Or as a library:

```python
from ts_anomaly import (
    TimeSeriesGenerator, TransformerAutoencoder, AnomalyDetector,
    moving_zscore, roc_auc, point_adjusted_f1, scores_to_labels,
)
from ts_anomaly.train import train_autoencoder

series, labels = TimeSeriesGenerator(seed=7, trend_slope=0.0).generate(
    n=1500, n_episodes=10
)

ae = TransformerAutoencoder(window=24, d_model=12, d_ff=24)
train_autoencoder(ae, series[:720], epochs=20)  # normal data only

detector = AnomalyDetector(ae)
threshold = detector.fit_threshold(series[:720], quantile=0.99, smooth=5)
scores = detector.score(series, smooth=5)
pred = scores_to_labels(scores, threshold)

print(point_adjusted_f1(labels, pred))  # episode-level F1
print(roc_auc(scores, labels))
```

## How detection works

1. **Generate** (`ts_anomaly/generator.py`): daily + weekly seasonal
   harmonics, a slow trend, and Gaussian noise, with four planted
   episode types: point spikes, level shifts, collective anomalies
   (suppressed seasonality: plausible values, wrong shape), and
   contextual anomalies (daytime-sized peaks injected at night).
   Everything is seeded and reproducible; labels are point-level.
2. **Attend** (`ts_anomaly/attention.py`): scaled dot-product
   self-attention, learned positional encodings, and a 2-layer
   Transformer encoder with residual connections, layer normalization,
   and a feed-forward block, all with hand-written backpropagation.
   No numpy, no torch.
3. **Reconstruct** (`ts_anomaly/model.py`): the autoencoder is trained
   on normal windows only. The anomaly score for a timestep is its
   squared reconstruction error, averaged over every sliding window
   covering it, optionally smoothed.
4. **Train** (`ts_anomaly/train.py`): mini-batch SGD with gradient-norm
   clipping, a train/val split, and early stopping that restores the
   best checkpoint. Inputs are standardized with normal-data stats so
   the encoder trains in a well-conditioned space.
5. **Evaluate** (`ts_anomaly/evaluate.py`): point-wise
   precision/recall/F1, point-adjusted F1 (the field standard: flagging
   any point inside a true episode counts the whole episode as
   detected), and ROC-AUC, all from scratch.

## Why reconstruction beats fixed thresholds

A moving z-score asks "is this value extreme relative to its
neighbors?" A collective anomaly answers "no" at every point and still
represents a real incident. The Transformer asks a different question:
"can I reconstruct this window from what normal looks like?" A
flattened daily cycle or a peak at 3 AM reconstructs badly even though
no single value crosses a threshold. On the synthetic benchmark the
reconstruction detector beats the z-score baseline on point-adjusted F1
and AUC; see `examples/quickstart.py` output for the exact numbers on
your run.

## Roadmap

- [ ] Multivariate series (per-channel encoders with cross-channel attention)
- [ ] Informer-style sparse attention for long windows
- [ ] Streaming inference: incremental scoring without re-encoding the full window
- [ ] Learned threshold calibration instead of a fixed quantile

## Tests

```bash
python3 -m pytest tests/ -q
```

Covers: attention weights sum to 1, positional encodings differ per
position, gradient flow through the encoder, normal windows reconstruct
better than anomalous ones after training, point-adjust logic on
crafted labels, generator episode labels matching a fixed seed, the
z-score spike response, and ROC-AUC edge cases.

> **Disclaimer:** every series in this repo is synthetic. The generator
> produces seasonality, trend, and noise from a seeded RNG, and all
> anomalies are planted by the generator itself. Nothing here is real
> telemetry, and a model trained on synthetic data should not be
> deployed against production systems without retraining on real data.
