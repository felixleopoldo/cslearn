library("stagedtrees")
 # Generate all the outcomes in the order specified by the data labels

datafile <- snakemake@input$data
modelfile <- snakemake@input$st

# read data from csv
data <- read.csv(datafile, header = TRUE, check.names = FALSE)
# the data is only needed for the cardinalities
# we get it from the first row
cardinalities <- data[1, ]

# read the model from the rds file
model <- readRDS(modelfile)

spaces <- list()
i <- 1
for (c in cardinalities) {
    spaces[[colnames(data)[[i]]]] <- seq(c) - 1
    i <- i + 1
}
space <- rev(expand.grid(rev(spaces)))

## generate the full sample space
prob <- prob(model, space)
log_prob <- prob(model, space, log = TRUE)
df <- cbind(space, prob = prob, log_prob = log_prob)

#print(df)

write.csv(df, snakemake@output$full_distr, row.names = FALSE)