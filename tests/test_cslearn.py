import random

import numpy as np
import numpy.testing as npt
import pandas as pd

import cslearn
from cslearn import cstree as ct


def test_project_defines_author_and_version():
    assert hasattr(cslearn, "__author__")
    assert hasattr(cslearn, "__version__")


def test_predict():
    np.random.seed(22)
    random.seed(22)

    cards = [3, 2, 2, 3]

    t = ct.sample_cstree(cards, max_cvars=2, prob_cvar=0.5, prop_nonsingleton=1)
    t.sample_stage_parameters(alpha=2)
    t.sample(100)

    # empty partial observation — predict all variables unconditionally (1-row, no columns)
    result = t.predict(pd.DataFrame(index=[0]))
    assert result.values.tolist() == [[2, 0, 1, 1]]
    result_p = t.predict(pd.DataFrame(index=[0]), return_probs=True)
    assert result_p[t.labels].values.tolist() == [[2, 0, 1, 1]]
    assert abs(result_p["PROB"].iloc[0] - 0.11590809641032397) < 1e-10

    # complete observation — return as-is with PROB=1
    complete = pd.DataFrame({0: [2], 1: [0], 2: [1], 3: [1]})
    result_c = t.predict(complete, return_probs=True)
    assert result_c[t.labels].values.tolist() == [[2, 0, 1, 1]]
    assert result_c["PROB"].iloc[0] == 1.0

    # partial: one variable observed
    po1 = pd.DataFrame({0: [1]})
    result1 = t.predict(po1)
    assert result1.values.tolist() == [[0, 1, 1]]
    result1_p = t.predict(po1, return_probs=True)
    assert result1_p[[1, 2, 3]].values.tolist() == [[0, 1, 1]]
    assert abs(result1_p["PROB"].iloc[0] - 0.26674411494959) < 1e-10

    # partial: two variables observed (non-contiguous)
    po2 = pd.DataFrame({0: [1], 3: [2]})
    result2 = t.predict(po2)
    assert result2.values.tolist() == [[1, 1]]

    # partial: three variables observed
    po3 = pd.DataFrame({0: [1], 2: [0], 3: [2]})
    result3 = t.predict(po3)
    assert result3.values.tolist() == [[0]]

    # multiple rows
    po_multi = pd.DataFrame({0: [1, 2]})
    result_multi = t.predict(po_multi)
    assert len(result_multi) == 2

    # sparse data — shouldn't raise KeyError
    s = ct.sample_cstree(cards, max_cvars=2, prob_cvar=0.5, prop_nonsingleton=1)
    s.sample_stage_parameters(alpha=2)
    s.sample(35)
    s.predict(pd.DataFrame(columns=s.labels))


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


def test_df_to_cstree_roundtrip_structure():
    """to_df / df_to_cstree preserves cards, labels, and stage count per level."""
    np.random.seed(5)
    random.seed(5)
    t = ct.sample_cstree([2, 2, 2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
    t.sample_stage_parameters(alpha=1.0)

    t2 = ct.df_to_cstree(t.to_df())

    assert list(t2.cards) == list(t.cards)
    assert list(t2.labels) == list(t.labels)
    for level in t.stages:
        assert len(t2.stages[level]) == len(t.stages[level])


def test_df_to_cstree_roundtrip_distribution():
    """Writing probs with write_probs=True and reading back reproduces the joint distribution."""
    np.random.seed(7)
    random.seed(7)
    t = ct.sample_cstree([2, 2, 2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
    t.sample_stage_parameters(alpha=1.0)

    t2 = ct.df_to_cstree(t.to_df(write_probs=True))

    dist1 = t.to_joint_distribution()
    dist2 = t2.to_joint_distribution()
    npt.assert_allclose(dist1["prob"].values, dist2["prob"].values, atol=1e-10)


# ---------------------------------------------------------------------------
# Parameter estimation
# ---------------------------------------------------------------------------


def test_estimate_stage_parameters_bdeu():
    """BDeu-estimated stage parameters sum to 1 at every stage."""
    np.random.seed(3)
    random.seed(3)
    t = ct.sample_cstree([2, 2, 2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
    t.sample_stage_parameters(alpha=1.0)
    data = t.sample(300)

    t.estimate_stage_parameters(data, method="BDeu", alpha_tot=1.0)

    for level, stages in t.stages.items():
        if level < 0:
            continue
        for stage in stages:
            if stage.probs is not None:
                npt.assert_allclose(sum(stage.probs), 1.0, atol=1e-10)


def test_estimate_stage_parameters_mle():
    """MLE (alpha_tot=0) stage parameters sum to 1 at every stage."""
    np.random.seed(3)
    random.seed(3)
    t = ct.sample_cstree([2, 2, 2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
    t.sample_stage_parameters(alpha=1.0)
    data = t.sample(500)

    t.estimate_stage_parameters(data, method="BDeu", alpha_tot=0)

    for level, stages in t.stages.items():
        if level < 0:
            continue
        for stage in stages:
            if stage.probs is not None:
                npt.assert_allclose(sum(stage.probs), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Joint distribution / pmf / pmf_log
# ---------------------------------------------------------------------------


def test_to_joint_distribution_sums_to_one():
    np.random.seed(2)
    random.seed(2)
    t = ct.sample_cstree([2, 2, 2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
    t.sample_stage_parameters(alpha=1.0)
    npt.assert_allclose(t.to_joint_distribution()["prob"].sum(), 1.0, atol=1e-10)


def test_to_joint_distribution_nonnegative():
    np.random.seed(2)
    random.seed(2)
    t = ct.sample_cstree([2, 2, 2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
    t.sample_stage_parameters(alpha=1.0)
    assert (t.to_joint_distribution()["prob"].values >= 0).all()


def test_pmf_matches_joint_distribution():
    """pmf(outcome) agrees with to_joint_distribution for every outcome."""
    np.random.seed(2)
    random.seed(2)
    t = ct.sample_cstree([2, 2, 2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
    t.sample_stage_parameters(alpha=1.0)
    dist = t.to_joint_distribution()
    for _, row in dist.iterrows():
        outcome = [int(row[l]) for l in t.labels]
        npt.assert_allclose(t.pmf(outcome), row["prob"], atol=1e-12)


def test_pmf_log_consistent_with_pmf():
    """pmf_log(outcome) == log(pmf(outcome)) for all positive-probability outcomes."""
    np.random.seed(2)
    random.seed(2)
    t = ct.sample_cstree([2, 2, 2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
    t.sample_stage_parameters(alpha=1.0)
    dist = t.to_joint_distribution()
    for _, row in dist.iterrows():
        outcome = [int(row[l]) for l in t.labels]
        p = t.pmf(outcome)
        if p > 0:
            npt.assert_allclose(t.pmf_log(outcome), np.log(p), atol=1e-12)


def test_pmf_label_order_permutation():
    """pmf with a permuted label_order gives the same probability as default order."""
    np.random.seed(2)
    random.seed(2)
    t = ct.sample_cstree([2, 2, 3], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
    t.sample_stage_parameters(alpha=1.0)
    outcome = [0, 1, 2]
    rev_labels = list(reversed(t.labels))
    outcome_rev = [outcome[t.labels.index(l)] for l in rev_labels]
    npt.assert_allclose(t.pmf(outcome_rev, label_order=rev_labels), t.pmf(outcome), atol=1e-12)
