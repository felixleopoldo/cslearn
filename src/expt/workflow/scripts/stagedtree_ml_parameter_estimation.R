library(stagedtrees)

# stagedtrees fits parameters during structure learning, so "parameter estimation"
# here is a pass-through: we simply re-save the already-fitted model.
data <- read.csv(snakemake@input$data, header = TRUE, check.names = FALSE)
model <- readRDS(snakemake@input$model)

saveRDS(model, snakemake@output$param_est)
