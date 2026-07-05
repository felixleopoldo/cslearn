import random

import numpy as np

from cslearn import cstree as ct
from cslearn.evaluate import kl_divergence, shd_ldag


def test_kl_divergence():
    np.random.seed(22)
    random.seed(22)

    cards = [3, 2, 2, 3]

    # KL-divergence is 0 when models are identical
    t = ct.sample_cstree(cards, max_cvars=2, prob_cvar=0.5, prop_nonsingleton=1)
    t.sample_stage_parameters(alpha=2)

    t.sample(1000)

    assert kl_divergence(t, t) == 0

    # KL-divergence is positive nonzero whet models are different
    e = ct.sample_cstree(cards, max_cvars=2, prob_cvar=0.5, prop_nonsingleton=1)
    e.sample_stage_parameters(alpha=2)

    e.sample(1000)
    assert kl_divergence(e, t) > 0

    # Conditional probabilities exist even when all outcomes haven't
    # been observed
    s = ct.sample_cstree(cards, max_cvars=2, prob_cvar=0.5, prop_nonsingleton=1)
    s.sample_stage_parameters(alpha=2)

    s.sample(35)
    kl_divergence(s, t)  # shouldn't raise KeyError


def test_shd_ldag():
    np.random.seed(42)
    random.seed(42)

    cards = [3, 2, 2, 3]
    t = ct.sample_cstree(cards, max_cvars=2, prob_cvar=0.5, prop_nonsingleton=1)

    # identity
    assert shd_ldag(t, t) == 0

    # positivity: two independently sampled trees should differ
    np.random.seed(7)
    random.seed(7)
    e = ct.sample_cstree(cards, max_cvars=2, prob_cvar=0.5, prop_nonsingleton=1)
    assert shd_ldag(e, t) > 0

    # non-negativity always holds (reversals ≤ symmetric difference size)
    assert shd_ldag(e, t) >= 0

    # reversal counts as 1: manually patch to_LDAG on minimal trees
    import networkx as nx
    from cslearn.ldag import LDAG as LDAGClass
    t1 = ct.sample_cstree([2, 2], max_cvars=1, prob_cvar=0, prop_nonsingleton=1)
    t2 = ct.sample_cstree([2, 2], max_cvars=1, prob_cvar=0, prop_nonsingleton=1)
    t1.to_LDAG = lambda: LDAGClass(nx.DiGraph([(0, 1)]))
    t2.to_LDAG = lambda: LDAGClass(nx.DiGraph([(1, 0)]))
    assert shd_ldag(t1, t2) == 1  # one reversal, not 2
