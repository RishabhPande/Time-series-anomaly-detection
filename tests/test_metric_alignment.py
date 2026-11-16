"""Metric inputs must refer to the same set of timesteps."""
import itertools

import pytest

from ts_anomaly.evaluate import (
    point_adjust, point_adjusted_f1, precision_recall_f1, roc_auc,
)


@pytest.mark.parametrize("metric", [precision_recall_f1, point_adjust, point_adjusted_f1])
@pytest.mark.parametrize("truth,pred", [
    ([1, 0, 1], [1, 0]),
    ([1, 0], [1, 0, 1]),
    ([0, 1, 1], [0, 1]),
    ([], [1]),
    ([1], []),
])
def test_binary_metrics_reject_mismatched_timesteps(metric, truth, pred):
    original_truth, original_pred = list(truth), list(pred)
    with pytest.raises(ValueError, match="same length"):
        metric(truth, pred)
    assert truth == original_truth
    assert pred == original_pred


@pytest.mark.parametrize("scores,truth", [
    ([0.9, 0.1], [1, 0, 1]),
    ([0.9, 0.1, 0.0], [1, 0]),
    ([], [1]),
    ([0.5], []),
])
def test_auc_rejects_mismatched_timesteps(scores, truth):
    with pytest.raises(ValueError, match="same length"):
        roc_auc(scores, truth)


def test_matched_empty_inputs_keep_existing_conventions():
    assert precision_recall_f1([], []) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert point_adjust([], []) == []
    assert point_adjusted_f1([], []) == precision_recall_f1([], [])
    assert roc_auc([], []) == 0.5


def test_single_class_auc_keeps_existing_neutral_convention():
    assert roc_auc([0.1, 0.9], [0, 0]) == 0.5
    assert roc_auc([0.1, 0.9], [1, 1]) == 0.5


def test_matched_episode_adjustment_preserves_inputs_and_false_positives():
    truth = [1, 1, 0, 0, 1, 1]
    pred = [0, 1, 0, 1, 0, 0]
    assert point_adjust(truth, pred) == [1, 1, 0, 1, 0, 0]
    assert truth == [1, 1, 0, 0, 1, 1]
    assert pred == [0, 1, 0, 1, 0, 0]


def test_auc_matches_pairwise_definition_for_all_small_tied_inputs():
    for labels in itertools.product([0, 1], repeat=4):
        for scores in itertools.product([0.0, 0.5, 1.0], repeat=4):
            pos = [s for s, t in zip(scores, labels) if t]
            neg = [s for s, t in zip(scores, labels) if not t]
            expected = (sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
                        / (len(pos) * len(neg))) if pos and neg else 0.5
            assert roc_auc(list(scores), list(labels)) == expected
