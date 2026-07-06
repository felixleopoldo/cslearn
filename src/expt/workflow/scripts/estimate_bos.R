library(stagedtrees)

# Data format: first row contains variable cardinalities; remaining rows are observations.
data <- read.csv(snakemake@input$data, header = TRUE, check.names = FALSE)
cardinalities <- data[1, ]
data <- data[-1, ]
df <- as.data.frame(data)

# BOS: optimal-order search using stages_bhc as the per-order scoring function.
start_time <- Sys.time()
model <- stagedtrees::search_best(df, alg = stages_bhc)
end_time <- Sys.time()

saveRDS(model, snakemake@output$st)

total_time <- end_time - start_time
time <- data.frame(
  method = "bos",
  time = total_time,
  total_time = total_time,
  n = snakemake@wildcards$cstree_data_n,
  seed = snakemake@wildcards$seed,
  p = snakemake@wildcards$cstree_p
)
write.csv(time, snakemake@output$time, row.names = FALSE, quote = FALSE)
