library(stagedtrees)

# read in the data

datafile <- snakemake@input$data
modelfile <- snakemake@input$model
#print(datafile)
#print(modelfile)

# read data from csv
data <- read.csv(datafile, header = TRUE, check.names = FALSE)
#print(head(data))


# read the model from the rds file
model <- readRDS(modelfile)

# here we should estimate the distribution, but I think its already estimated
#print(model)

saveRDS(model, snakemake@output$param_est)

