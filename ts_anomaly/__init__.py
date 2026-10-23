"""Transformer-based time-series anomaly detection, pure Python."""

from .attention import TransformerEncoder
from .baseline import moving_zscore
from .evaluate import (
    point_adjusted_f1,
    precision_recall_f1,
    roc_auc,
    scores_to_labels,
)
from .generator import TimeSeriesGenerator
from .model import AnomalyDetector, TransformerAutoencoder
from .train import make_windows, train_autoencoder

__all__ = [
    "TimeSeriesGenerator",
    "TransformerEncoder",
    "TransformerAutoencoder",
    "AnomalyDetector",
    "train_autoencoder",
    "make_windows",
    "moving_zscore",
    "scores_to_labels",
    "precision_recall_f1",
    "point_adjusted_f1",
    "roc_auc",
]

__version__ = "0.1.0"
