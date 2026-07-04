library(stagedtrees)

# Compute LDAG edges from a staged event tree.
#
# stages(model)[[v]] stores one stage label per position at level l (2^(l-1)
# entries). Positions are in DFS order: first variable is highest bit, so two
# positions differing only in ancestor j (0-indexed) differ at bit (l-2-j).
# Vectorised bitwise XOR replaces string-splitting and is ~1000x faster.
# Fast-path: skip variables where all positions share one stage.
compute_ldag_edges <- function(model) {
  vars  <- sevt_varnames(model)
  n     <- length(vars)
  stg   <- stages(model)
  ldag_edges <- character(0)

  for (l in seq(2, n)) {
    var_l <- vars[l]
    sv    <- stg[[var_l]]
    np    <- length(sv)

    if (length(unique(sv)) == 1L) next   # uniform staging: no context vars

    for (jp in seq(0L, l - 2L)) {
      bit <- bitwShiftL(1L, l - 2L - jp)
      i0  <- which(bitwAnd(seq_len(np) - 1L, bit) == 0L) - 1L
      i1  <- bitwXor(i0, bit)
      if (any(sv[i0 + 1L] != sv[i1 + 1L]))
        ldag_edges <- c(ldag_edges, paste(vars[jp + 1L], var_l))
    }
  }

  ldag_edges
}

model   <- readRDS(snakemake@input$est)
est_set <- compute_ldag_edges(model)

true_df  <- read.csv(snakemake@input$true_ldag, colClasses = "character")
true_set <- paste(true_df$u, true_df$v)

if (length(est_set) > 0) {
  est_rev   <- sapply(strsplit(est_set, " "), function(p) paste(p[2], p[1]))
  reversals <- sum(est_rev %in% true_set)
} else {
  reversals <- 0L
}
sym_diff <- length(union(est_set, true_set)) - length(intersect(est_set, true_set))
shd_val  <- sym_diff - reversals

write.csv(
  data.frame(
    method = snakemake@params$method,
    shd    = shd_val,
    seed   = snakemake@wildcards$seed,
    p      = snakemake@wildcards$cstree_p,
    n      = snakemake@wildcards$cstree_data_n
  ),
  snakemake@output[[1]],
  row.names = FALSE,
  quote     = FALSE
)
