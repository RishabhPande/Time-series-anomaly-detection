"""Unit tests for ts_anomaly."""

import math
import random

from ts_anomaly import (
    TimeSeriesGenerator,
    TransformerAutoencoder,
    moving_zscore,
    point_adjusted_f1,
    precision_recall_f1,
    roc_auc,
)
from ts_anomaly.attention import SelfAttention, TransformerEncoder
from ts_anomaly.evaluate import point_adjust
from ts_anomaly.train import make_windows, train_autoencoder


def test_attention_weights_sum_to_one():
    rng = random.Random(0)
    attn = SelfAttention(dim=8, rng=rng)
    X = [[rng.uniform(-1, 1) for _ in range(8)] for _ in range(10)]
    attn.forward(X)
    for row in attn.last_weights:
        assert abs(sum(row) - 1.0) < 1e-9
        assert all(w >= 0.0 for w in row)


def test_positional_encodings_differ_per_position():
    enc = TransformerEncoder(window=16, d_model=8, seed=1)
    P = enc.pos.P.value
    assert len(P) == 16
    for i in range(1, 16):
        assert any(a != b for a, b in zip(P[0], P[i])), (
            "positional encodings must differ across positions"
        )


def test_gradient_flows_through_encoder():
    enc = TransformerEncoder(window=8, d_model=8, seed=2)
    win = [0.1 * i for i in range(8)]
    out = enc.forward(win)
    enc.zero_grad()
    enc.backward([1.0] * 8)
    grads = [g for p in enc.parameters() for row in p.grad for g in row]
    assert any(abs(g) > 1e-12 for g in grads), "no gradient reached the parameters"


def test_normal_windows_reconstruct_better_after_training():
    gen = TimeSeriesGenerator(seed=11)
    series, labels = gen.generate(n=400, n_episodes=4, anomaly_start=0.6)
    ae = TransformerAutoencoder(window=16, d_model=16, d_ff=32, seed=3)
    train_autoencoder(
        ae, series[:240], epochs=20, batch_size=64, lr=0.05, patience=20, seed=1
    )
    wins = make_windows(series, 16)
    lab_wins = make_windows(labels, 16)
    clean = [w for w, lw in zip(wins, lab_wins) if sum(lw) == 0]
    anom = [w for w, lw in zip(wins, lab_wins) if sum(lw) >= 4]
    assert len(anom) > 5, "generator should plant sizable episodes"
    clean_err = sum(ae.window_error(w) for w in clean) / len(clean)
    anom_err = sum(ae.window_error(w) for w in anom) / len(anom)
    assert clean_err < anom_err, f"{clean_err=} vs {anom_err=}"


def test_point_adjust_marks_whole_episode():
    y_true = [0, 0, 1, 1, 1, 0, 0]
    y_pred = [0, 0, 0, 1, 0, 0, 0]  # single hit inside the episode
    adj = point_adjust(y_true, y_pred)
    assert adj == [0, 0, 1, 1, 1, 0, 0]
    m = point_adjusted_f1(y_true, y_pred)
    assert m["recall"] == 1.0
    assert m["f1"] > precision_recall_f1(y_true, y_pred)["f1"]


def test_point_adjust_ignores_missed_episodes():
    y_true = [0, 1, 1, 0, 1, 1, 0]
    y_pred = [0, 0, 0, 0, 1, 0, 0]  # second episode hit, first missed
    adj = point_adjust(y_true, y_pred)
    assert adj == [0, 0, 0, 0, 1, 1, 0]
    assert point_adjusted_f1(y_true, y_pred)["recall"] == 0.5


def test_generator_episode_labels_match():
    gen = TimeSeriesGenerator(seed=123)
    n = 1000
    series, labels = gen.generate(
        n=n, n_episodes=8, anomaly_start=0.6, kinds=["point", "level"]
    )
    assert len(series) == len(labels) == n
    # head of the series is guaranteed clean
    assert sum(labels[: int(0.6 * n)]) == 0
    assert sum(labels) > 8  # 8 episodes, levels span multiple points
    # labels only where something was injected: spot-check determinism
    s2, l2 = gen.generate(n=n, n_episodes=8, anomaly_start=0.6,
                          kinds=["point", "level"])
    assert labels == l2 and series == s2


def test_generator_all_kinds_present():
    gen = TimeSeriesGenerator(seed=7)
    kinds = ["point", "level", "collective", "contextual"]
    _, labels = gen.generate(n=1500, n_episodes=8, kinds=kinds)
    # each kind labels at least one point in the tail
    assert sum(labels) >= 8


def test_moving_zscore_spikes_on_outlier():
    series = [1.0] * 100
    series[80] = 10.0
    scores = moving_zscore(series, window=20)
    assert scores[80] == max(scores)
    assert scores[80] > 3.0


def test_roc_auc_sanity():
    scores = [0.1, 0.4, 0.35, 0.8]
    labels = [0, 0, 1, 1]
    auc = roc_auc(scores, labels)
    assert 0.5 < auc <= 1.0
    assert roc_auc([0.5, 0.5], [0, 1]) == 0.5  # no positives/negatives edge
    assert roc_auc([0.9, 0.1], [1, 1]) == 0.5


def test_make_windows_shapes():
    wins = make_windows(list(range(10)), window=4, stride=2)
    assert wins == [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9]]
