# import pytest
import random

import numpy as np
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
