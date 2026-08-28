"""Tests for the structure learning pipeline: find_optimal_cstree, CStree.fit,
all_stagings with poss_cvars, and the causallearn bridge functions."""

import random

import numpy as np
import pytest

import cslearn.learning as ctl
import cslearn.scoring as sc
from cslearn import cstree as ct

# ---------------------------------------------------------------------------
# find_optimal_cstree / CStree.fit
# ---------------------------------------------------------------------------


def test_find_optimal_cstree_returns_valid_cstree():
    """find_optimal_cstree returns a CStree with the correct cardinalities and labels."""
    np.random.seed(10)
    random.seed(10)
    t = ct.sample_cstree([2, 2, 2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
    t.sample_stage_parameters(alpha=1.0)
    data = t.sample(300)

    opt = ctl.find_optimal_cstree(data, max_cvars=1)

    assert isinstance(opt, ct.CStree)
    assert opt.cards == t.cards
    assert opt.labels == t.labels


def test_find_optimal_cstree_has_stages():
    """The learned tree has a non-empty staging at every level."""
    np.random.seed(12)
    random.seed(12)
    t = ct.sample_cstree([2, 2, 2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
    t.sample_stage_parameters(alpha=1.0)
    data = t.sample(400)

    opt = ctl.find_optimal_cstree(data, max_cvars=2)

    assert any(len(stages) > 0 for stages in opt.stages.values())


def test_fit_updates_self_and_returns_self():
    """CStree.fit learns a structure in place and returns self."""
    np.random.seed(11)
    random.seed(11)
    t = ct.sample_cstree([2, 2, 2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
    t.sample_stage_parameters(alpha=1.0)
    data = t.sample(300)

    fresh = ct.CStree(t.cards, labels=t.labels)
    result = fresh.fit(data, max_cvars=1, gibbs_samples=200)

    assert result is fresh
    assert isinstance(fresh.stages, dict)
    assert fresh.cards == t.cards


# ---------------------------------------------------------------------------
# all_stagings with poss_cvars
# ---------------------------------------------------------------------------


def test_all_stagings_poss_cvars_reduces_count():
    """Restricting splittable coordinates reduces the number of stagings."""
    cards = [2, 2, 2]
    level = 1
    all_count = sum(1 for _ in ctl.all_stagings(cards, level, max_cvars=2))
    restricted = sum(1 for _ in ctl.all_stagings(cards, level, max_cvars=2, poss_cvars=[0]))
    assert restricted < all_count


def test_all_stagings_poss_cvars_respects_restriction():
    """With poss_cvars=[0], no stage fixes coordinate 1 as a context variable."""
    cards = [2, 2, 2]
    for staging in ctl.all_stagings(cards, level=1, max_cvars=2, poss_cvars=[0]):
        for stage in staging:
            if len(stage.list_repr) > 1:
                assert not isinstance(stage.list_repr[1], int), (
                    f"coord 1 used as context despite poss_cvars=[0]: {stage}"
                )


def test_all_stagings_poss_cvars_empty_gives_trivial_staging():
    """Empty poss_cvars: only the all-free (singleton) staging is returned."""
    stagings = list(ctl.all_stagings([2, 2, 2], level=1, max_cvars=2, poss_cvars=[]))
    assert len(stagings) == 1


# ---------------------------------------------------------------------------
# causallearn bridge functions
# ---------------------------------------------------------------------------


def test_causallearn_graph_to_posscvars_output_shape():
    """causallearn_graph_to_posscvars maps every variable label to a list of parent labels."""
    from causallearn.search.ConstraintBased.PC import pc

    np.random.seed(1)
    random.seed(1)
    t = ct.sample_cstree([2, 2, 2, 2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
    t.sample_stage_parameters(alpha=1.0)
    data = t.sample(300)
    labels = list(t.labels)

    cg = pc(data.values[1:].astype(float), alpha=0.05, show_progress=False)
    poss_cvars = ctl.causallearn_graph_to_posscvars(cg, labels, alg="pc")

    assert set(poss_cvars.keys()) == set(labels)
    for var_parents in poss_cvars.values():
        assert isinstance(var_parents, list)
        assert all(p in labels for p in var_parents)


def test_causallearn_graph_to_dag_output_shape():
    """causallearn_graph_to_dag returns a square binary adjacency DataFrame."""
    pytest.importorskip("pgmpy")
    from causallearn.search.ConstraintBased.PC import pc

    np.random.seed(1)
    random.seed(1)
    t = ct.sample_cstree([2, 2, 2, 2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
    t.sample_stage_parameters(alpha=1.0)
    data = t.sample(300)
    labels = list(t.labels)

    cg = pc(data.values[1:].astype(float), alpha=0.05, show_progress=False)
    dag_df = ctl.causallearn_graph_to_dag(cg, labels, alg="pc")

    assert set(dag_df.columns) == set(labels)
    assert set(dag_df.index) == set(labels)
    assert set(dag_df.values.flatten()).issubset({0, 1})


def test_causallearn_posscvars_used_in_scoring():
    """poss_cvars from PC can be passed into order_score_tables and produce a valid CStree."""
    from causallearn.search.ConstraintBased.PC import pc

    np.random.seed(2)
    random.seed(2)
    t = ct.sample_cstree([2, 2, 2, 2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
    t.sample_stage_parameters(alpha=1.0)
    data = t.sample(300)
    labels = list(t.labels)

    cg = pc(data.values[1:].astype(float), alpha=0.05, show_progress=False)
    poss_cvars = ctl.causallearn_graph_to_posscvars(cg, labels, alg="pc")

    score_table, context_scores, _ = sc.order_score_tables(data, max_cvars=1, alpha_tot=1.0, poss_cvars=poss_cvars)
    opt_order, _ = ctl._find_optimal_order(score_table)
    opt_tree = ctl._optimal_cstree_given_order(opt_order, context_scores)

    assert isinstance(opt_tree, ct.CStree)
    assert opt_tree.cards == t.cards
