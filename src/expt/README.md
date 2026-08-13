# CSlearn experiments

Snakemake workflow for the simulation experiments and figures in the paper.
All commands should be run from this directory (`src/expt/`).

## Requirements

- [Snakemake](https://snakemake.readthedocs.io/) ≥ 7.0
- [Apptainer](https://apptainer.org/) (for container execution; includes `docker://` image support)

## Directory layout

```
src/expt/
├── workflow/
│   ├── Snakefile       # main workflow (auto-discovered by snakemake)
│   ├── scripts/        # one script per rule + aggregation/plot scripts
│   └── envs/           # Dockerfiles for the benchmark containers
└── config/             # workflow configuration (to be populated)
```

Output goes to `results/` (gitignored, except the aggregated CSVs listed below).

## Reproducing the paper figures

Eleven aggregated CSVs are committed to the repository:

| File | Content |
|------|---------|
| `results/kl_divergence_2a.csv` | KL divergence: CSlearn+PC vs PC+DAG baseline |
| `results/kl_divergence_2b.csv` | KL divergence: CSlearn+GRaSP vs GRaSP+DAG baseline |
| `results/kl_divergence_2c.csv` | KL divergence: CSlearn+GRaSP vs BOS vs GRaSP+BHC |
| `results/time_3a.csv` | Runtime: all methods, p=5–20 |
| `results/time_3b.csv` | Runtime: scalable methods, p=10–500 |
| `results/shd_a.csv` | SHD: CSlearn+PC vs PC+DAG |
| `results/shd_b.csv` | SHD: CSlearn+GRaSP vs GRaSP+DAG |
| `results/shd_c.csv` | SHD: CSlearn+GRaSP vs BOS vs GRaSP+BHC |
| `results/shd_scale.csv` | SHD: scalable methods, p=10–500 |
| `results/sensitivity_shd.csv` | Sensitivity to β misspecification: SHD |
| `results/sensitivity_kl.csv` | Sensitivity to β misspecification: KL divergence |

Regenerating all figures requires only Snakemake and Apptainer — no data
downloads or recomputation:

```bash
snakemake kl_plots_2a kl_plots_2b kl_plots_2c time_plots_3a time_plots_3b \
  shd_plot_a shd_plot_b shd_plot_c shd_plots_scale sensitivity_shd_plot sensitivity_kl_plot \
  --use-apptainer --cores 1 \
  --allowed-rules kl_plots_2a kl_plots_2b kl_plots_2c time_plots_3a time_plots_3b \
    shd_plot_a shd_plot_b shd_plot_c shd_plots_scale sensitivity_shd_plot sensitivity_kl_plot
```

PDFs appear in `results/`.

(`--allowed-rules` prevents Snakemake from tracing the full upstream
simulation DAG when only the aggregated CSVs are available locally.)

## Full recomputation from scratch

Requires Apptainer and hundreds of CPU-hours.

```bash
snakemake all --use-apptainer --cores N
```

Dry run to preview what would be computed:

```bash
snakemake all --use-apptainer -n
```

## Recomputation from saved intermediates

The full set of precomputed results (~5.9 GB) is archived on Zenodo:
[https://doi.org/10.5281/zenodo.21198084](https://doi.org/10.5281/zenodo.21198084)

Downloading restores all intermediate files so that `snakemake all` skips
the expensive recomputation steps:

```bash
snakemake download_intermediates --cores 1
snakemake all --use-apptainer --cores N
```

This allows the workflow to be reproduced starting at any point in the
pipeline, without recomputing preceding steps.
