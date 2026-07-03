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

Output goes to `results/` (gitignored, except the 5 aggregated CSVs below).

## Reproducing the paper figures

The five aggregated CSVs (`results/kl_divergence_2{a,b,c}.csv`,
`results/time_3{a,b}.csv`) are committed to the repository. Regenerating
the figures from them requires only Snakemake and Apptainer — no data
downloads or recomputation:

```bash
snakemake kl_plots_2a kl_plots_2b kl_plots_2c time_plots_3a time_plots_3b \
  --use-apptainer --cores 1 \
  --allowed-rules kl_plots_2a kl_plots_2b kl_plots_2c time_plots_3a time_plots_3b
```

PDFs appear in `results/`.

(`--allowed-rules` prevents Snakemake from tracing the full upstream
simulation DAG when only the aggregated CSVs are available locally.)

## Full recomputation from scratch

Requires Apptainer and approximately 5.5 GB of disk space for intermediate
results. Recomputing from scratch takes hundreds of CPU-hours.

```bash
snakemake all --use-apptainer --cores N
```

Dry run to preview what would be computed:

```bash
snakemake all --use-apptainer -n
```

## Recomputation from saved intermediates

The full set of intermediate results is archived (update URL in
`workflow/Snakefile` after Zenodo deposit). Downloading restores all
intermediate files so that `snakemake all` skips recomputation:

```bash
snakemake download_intermediates --cores 1
snakemake all --use-apptainer --cores N
```
