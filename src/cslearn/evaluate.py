"""Evaluate estimated CStrees."""

from functools import reduce
from itertools import pairwise, product

import numpy as np
from scipy.special import rel_entr

from cslearn.cstree import CStree


def KL_divergence(df_distr1, df_distr2):
    """Calculate the KL divergence between two distributions using scipy rel_entr.
    df_distr2 is typically the true distribution and df_distr1 is the estimated distribution.
    """

    distr1 = df_distr1["prob"].values
    distr2 = df_distr2["prob"].values

    return np.sum(rel_entr(distr1, distr2))


def shd_edges(estimated_edges, true_edges) -> int:
    """Structural Hamming distance between two directed edge sets.

    Counts missing edges, extra edges, and reversed edges. A reversal counts
    as 1, not 2.
    """
    est_edges = set(estimated_edges)
    true_edges = set(true_edges)
    reversals = sum(1 for u, v in est_edges if (v, u) in true_edges)
    return len(est_edges.symmetric_difference(true_edges)) - reversals


def shd_ldag(estimated: CStree, true: CStree) -> int:
    """Structural Hamming distance between two CStrees via their LDAG representations.

    Counts missing edges, extra edges, and reversed edges between the LDAGs.
    A reversal counts as 1, not 2.
    """
    return shd_edges(estimated.to_LDAG().edges(), true.to_LDAG().edges())


def kl_divergence(estimated: CStree, true: CStree) -> float:
    """KL divergence D(estimated || true) between two CStree distributions.

    Args:
        estimated: The estimated CStree. Its labels must be a permutation of
            ``true.labels``.
        true: The true CStree. Its labels must be sorted (``true.labels ==
            sorted(true.labels)``); outcomes are enumerated in that order.

    Returns:
        float: KL divergence (non-negative; 0 iff distributions are identical).
    """
    factorized_outcomes = (range(card) for card in true.cards)
    outcomes = product(*factorized_outcomes)

    def _rel_entr_of_outcome(outcome):
        assert true.labels == sorted(true.labels)
        true_nodes = (outcome[:idx] for idx in range(true.p + 1))
        true_edges = pairwise(true_nodes)

        est_ordered_outcome = tuple(outcome[i] for i in estimated.labels)
        est_nodes = (est_ordered_outcome[:idx] for idx in range(true.p + 1))
        est_edges = pairwise(est_nodes)

        def _probs_map(zipped_edge):
            est_edge, true_edge = zipped_edge
            try:
                est = estimated.tree[est_edge[0]][est_edge[1]]["cond_prob"]
            except KeyError:
                stage = estimated.get_stage(est_edge[0])
                est = stage.probs[est_edge[1][-1]]
            try:
                tru = true.tree[true_edge[0]][true_edge[1]]["cond_prob"]
            except KeyError:
                stage = true.get_stage(true_edge[0])
                tru = stage.probs[true_edge[1][-1]]
            return est, tru

        zipped_edges = zip(est_edges, true_edges)
        zipped_probs = map(_probs_map, zipped_edges)

        def _probs_of_outcome(prev_pair, next_pair):
            return prev_pair[0] * next_pair[0], prev_pair[1] * next_pair[1]

        est_prob_outcome, true_prob_outcome = reduce(_probs_of_outcome, zipped_probs)
        return rel_entr(est_prob_outcome, true_prob_outcome)

    return sum(map(_rel_entr_of_outcome, outcomes))
