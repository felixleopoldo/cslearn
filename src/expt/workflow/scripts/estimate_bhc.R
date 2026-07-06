library(stagedtrees)
library(bnlearn)

# Convert a raw data matrix to a data frame with factor columns.
matrix_to_dataframe <- function(data, varnames) {
  datadf <- as.data.frame(data)
  colnames(datadf) <- varnames
  for (j in seq_along(varnames)) {
    datadf[[varnames[j]]] <- as.factor(datadf[[varnames[j]]])
  }
  datadf
}

# Data format: first row contains variable cardinalities; remaining rows are observations.
data <- read.csv(snakemake@input$data, header = TRUE, check.names = FALSE)
cardinalities <- data[1, ]
data <- data[-1, ]

start_time <- Sys.time()

# Read the DAG produced by the phase-1 algorithm (PC or GRaSP) and convert to bnlearn format.
dag_adjmat <- read.csv(snakemake@input$input_alg_dag, header = TRUE, check.names = FALSE)
dag_matrix <- as.matrix(dag_adjmat)
colnames(dag_matrix) <- colnames(data)

dag <- empty.graph(nodes = colnames(data))
amat(dag) <- dag_matrix

# Fit BN parameters, convert to staged event tree, then search for stages with BHC.
df <- matrix_to_dataframe(data, colnames(data))
dagfit <- bnlearn::bn.fit(dag, df)
model <- as_sevt(dagfit) |> sevt_fit(df, lambda = 0) |> stages_bhc()

end_time <- Sys.time()

saveRDS(model, snakemake@output$st)

# Total time includes the phase-1 DAG learning time.
bhc_time <- end_time - start_time
input_alg_time <- read.csv(snakemake@input$input_alg_time, header = TRUE, check.names = FALSE)
total_time <- bhc_time + input_alg_time$time

time <- data.frame(
  method = "bhc",
  time = bhc_time,
  total_time = total_time,
  n = snakemake@wildcards$cstree_data_n,
  seed = snakemake@wildcards$seed,
  p = snakemake@wildcards$cstree_p
)
write.csv(time, snakemake@output$time, row.names = FALSE, quote = FALSE)
