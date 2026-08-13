<div align="center">
  <a href="https://cslearn.readthedocs.io">
    <img src="images/cstree.png" alt="CStree logo">
  </a>
  <img src="images/minl_cont_dag_X1=0.png" alt="Minimal context DAG X1=0" width="40">
  <img src="images/minl_cont_dag_X2=0.png" alt="Minimal context DAG X2=0" width="80">
  <img src="images/minl_cont_dag_X3=0.png" alt="Minimal context DAG X3=0" width="80">

  <h3 align="center">CSlearn</h3>

  <p align="center">
    A Python library for context-specific causal graphical models.
    <br />
    <a href="https://cslearn.readthedocs.io"><strong>Docs »</strong></a>
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

See the [full installation instructions](https://cslearn.readthedocs.io/en/latest/install.html) for development setup.

## Quick start

```python
import pandas as pd
from cslearn import CStree, sample_cstree

# Sample a random CStree and simulate data
tree = sample_cstree([2, 2, 3, 2], max_cvars=2, prob_cvar=0.5)
tree.sample_stage_parameters(alpha=2.0)
data = tree.sample(500)

# Learn a CStree from data
learned = CStree().fit(data)

# Predict held-out observations
predictions = learned.predict(data.iloc[:5, :-1])
```

See the [example notebooks](https://cslearn.readthedocs.io) for walkthroughs covering CStree construction and visualization, structure learning with exact and Gibbs-sampler search, LDAG representations on the alarm and Sachs datasets, and prediction.

## Paper experiments

The simulation experiments and figures from the accompanying paper are in
`src/expt/`. Aggregated results (CSVs) are committed to the repository;
precomputed intermediates (~5.9 GB) are on Zenodo at
[https://doi.org/10.5281/zenodo.21198084](https://doi.org/10.5281/zenodo.21198084).
See [`src/expt/README.md`](src/expt/README.md) for reproduction instructions.

## Reference

If you use this package, please cite the accompanying paper:

> Rios, F. L., Markham, A. & Solus, L. (2024). Scalable Structure Learning for Sparse Context-Specific Systems. [arXiv:2402.07762](https://arxiv.org/abs/2402.07762)

```bibtex
@misc{rios2024scalablestructurelearningsparse,
      title={Scalable Structure Learning for Sparse Context-Specific Systems},
      author={Felix Leopoldo Rios and Alex Markham and Liam Solus},
      year={2024},
      eprint={2402.07762},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2402.07762},
}
```

## Contributing

Contributions are welcome. Please open an issue or pull request on
[GitHub](https://github.com/felixleopoldo/cslearn/issues).
