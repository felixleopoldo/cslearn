# CSlearn experiments

Snakemake workflow for the simulation experiments and figures in the paper.
All commands should be run from this directory (`src/expt/`).

## Requirements

- [Snakemake](https://snakemake.readthedocs.io/) ≥ 7.0
- Docker (for full pipeline runs; not needed for plot-only reproduction)
- `curl` (for downloading pre-computed results)

## Directory layout

```
src/expt/
├── workflow/
│   ├── Snakefile       # main workflow (auto-discovered by snakemake)
│   ├── scripts/        # one script per rule + aggregation/plot scripts
│   └── envs/           # Dockerfiles for the benchmark containers
└── config/             # workflow configuration (to be populated)
```

Output goes to `results/` (gitignored).

## Reproducing the paper figures

### Option A — from pre-computed results (fast, no Docker)

Download the aggregated CSVs and PDFs (78 KB) and regenerate figures:

```bash
snakemake download_results
snakemake kl_plots_2a kl_plots_2b kl_plots_2c time_plots_3a time_plots_3b
```

PDFs appear in `results/`.

### Option B — full recomputation (requires Docker, ~5.5 GB, hours of compute)

```bash
snakemake all
```

This generates all intermediate results from scratch inside Docker containers
(`bpimages/cslearnbenchmarks:1.3.1-amd64` for Python rules,
`bpimages/stagedtrees:2.3.0` for R/stagedtrees rules).

Dry run to preview what would be computed:

```bash
snakemake all -n
```
