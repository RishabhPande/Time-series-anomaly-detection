"""Evaluation metrics from scratch.

Point-wise precision/recall/F1, the field-standard point-adjusted F1
(Xu et al.: if any point inside a true anomaly episode is flagged, the
whole episode counts as detected), and ROC-AUC via the rank method.
"""

from __future__ import annotations


def scores_to_labels(scores: list[float], threshold: float) -> list[int]:
    return [1 if s >= threshold else 0 for s in scores]


def precision_recall_f1(
    y_true: list[int], y_pred: list[int]
) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _episodes(y_true: list[int]) -> list[tuple[int, int]]:
    """Contiguous runs of 1s as (start, end-exclusive) pairs."""
    eps, start = [], None
    for i, v in enumerate(y_true + [0]):
        if v == 1 and start is None:
            start = i
        elif v == 0 and start is not None:
            eps.append((start, i))
            start = None
    return eps


def point_adjust(y_true: list[int], y_pred: list[int]) -> list[int]:
    """Expand flags: any hit inside a true episode flags the whole episode."""
    adj = list(y_pred)
    for s, e in _episodes(y_true):
        if any(y_pred[s:e]):
            for i in range(s, e):
                adj[i] = 1
    return adj


def point_adjusted_f1(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return precision_recall_f1(y_true, point_adjust(y_true, y_pred))


def roc_auc(scores: list[float], y_true: list[int]) -> float:
    """Area under the ROC curve via the Mann-Whitney rank statistic."""
    pos = [s for s, t in zip(scores, y_true) if t == 1]
    neg = [s for s, t in zip(scores, y_true) if t == 0]
    if not pos or not neg:
        return 0.5
    ranked = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    # average ranks for ties
    rank_sum, i, n = 0.0, 0, len(ranked)
    while i < n:
        j = i
        while j < n and ranked[j][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(lbl for _, lbl in ranked[i:j])
        i = j
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
