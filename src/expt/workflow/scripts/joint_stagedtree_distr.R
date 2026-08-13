library(stagedtrees)

# Data is read only to extract variable cardinalities from the first row.
data <- read.csv(snakemake@input$data, header = TRUE, check.names = FALSE)
cardinalities <- data[1, ]

model <- readRDS(snakemake@input$st)

# Build the full outcome space respecting variable order from the data columns.
spaces <- list()
i <- 1
for (c in cardinalities) {
  spaces[[colnames(data)[[i]]]] <- seq(c) - 1
  i <- i + 1
}
# expand.grid reverses column order, so we reverse before and after to preserve it.
space <- rev(expand.grid(rev(spaces)))

prob <- prob(model, space)
log_prob <- prob(model, space, log = TRUE)
df <- cbind(space, prob = prob, log_prob = log_prob)

write.csv(df, snakemake@output$full_distr, row.names = FALSE)
