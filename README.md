<div align="center">
  <a href="docs/_build/html/index.html">
    <img src="images/cstree.png" alt="CStree logo">
  </a>
  <img src="images/minl_cont_dag_X1=0.png" alt="Minimal context DAG X1=0" width="40">
  <img src="images/minl_cont_dag_X2=0.png" alt="Minimal context DAG X2=0" width="80">
  <img src="images/minl_cont_dag_X3=0.png" alt="Minimal context DAG X3=0" width="80">

  <h3 align="center">CSlearn</h3>

  <p align="center">
    A Python library for context-specific causal graphical models.
    <br />
    <a href="docs/_build/html/index.html"><strong>Docs »</strong></a>
  </p>
</div>

## Overview

`cslearn` is a Python package for **CStree models**—a family of graphical causal models for multivariate discrete data that encode context-specific independence (CSI). CStrees generalize DAG models while remaining tractable.

The package implements **CSlearn**, a three-phase structure-learning algorithm:
1. DAG pre-screening via PC or GRaSP to restrict candidate parent sets
2. Order MCMC (Gibbs sampler) over topological orderings
3. Exact staging search under a sparsity bound

## Installation

`cslearn` requires [graphviz](https://graphviz.org/download/) to be installed on your system.

On Debian/Ubuntu/Linux:

```bash
sudo apt install graphviz libgraphviz-dev pkg-config
```

On macOS (Homebrew):

```bash
brew install graphviz
```

On Windows, install graphviz from https://graphviz.org/download/ and ensure it is on your `PATH`.

Then install the package:

```bash
pip install cslearn
```

See the [full installation instructions](docs/_build/html/install.html) for development setup.

## Quick start

```python
from cslearn import CStree, sample_cstree

# Sample a random CStree and simulate data
tree = sample_cstree([2, 2, 3, 2], max_cvars=2, prob_cvar=0.5)
tree.sample_stage_parameters(alpha=2.0)
data = tree.sample(500)

# Learn a CStree from data
learned = CStree().fit(data)

# Predict the last variable for five held-out observations
# (row 0 of `data` holds the cardinalities, so skip it)
predictions = learned.predict(data.iloc[1:6, :-1])
```

See the [example notebooks](docs/_build/html/source/examples.html) for walkthroughs covering CStree construction and visualization, structure learning with exact and Gibbs-sampler search, LDAG representations on the alarm and Sachs datasets, and prediction.

## Paper experiments

The simulation experiments and figures from the accompanying paper are in
`src/expt/`. Aggregated results (CSVs) are committed to the repository;
precomputed intermediates (~5.9 GB) can be fetched with `snakemake
download_intermediates`. See [`src/expt/README.md`](src/expt/README.md) for
reproduction instructions.

## Reference

Citation withheld for anonymous review.

## Contributing

Contributions are welcome. Please open an issue or pull request on
[GitHub](https://github.com/felixleopoldo/cslearn/issues).
