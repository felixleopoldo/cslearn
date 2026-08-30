================================================================================
CSlearn
================================================================================

.. figure:: _static/fig1_demo.png
    :align: center

    A CStree on four binary random variables.

``cslearn`` is a Python package for **CStree models** :footcite:p:`duarte2021representation`
:footcite:p:`rios2024scalable` -- a family of graphical causal models for
multivariate discrete data that encode context-specific independence (CSI).
CStrees generalize DAG models while remaining tractable.

The package implements **CSlearn**, a three-phase structure-learning algorithm:
DAG pre-screening (PC or GRaSP) to restrict candidate parent sets, order MCMC
over topological orderings, and an exact staging search under a sparsity bound.

See :doc:`install` to get started and the project README for a quick-start
example. The paper experiments and precomputed results (~5.9 GB) are on
`Zenodo <https://doi.org/10.5281/zenodo.21198084>`_.

.. toctree::
   :maxdepth: 5
   :hidden:
   :name: Getting started
   :caption: Getting started

   Installation <install.rst>
   Reference <source/cslearn.rst>
   Example notebooks <source/examples.rst>

.. toctree::
   :maxdepth: 2
   :hidden:
   :name: Project info
   :caption: Project info

   License <license.rst>

.. rubric:: References

.. footbibliography::
