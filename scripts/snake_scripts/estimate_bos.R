library(stagedtrees)

datafile <- snakemake@input$data

# read data from csv
data <- read.csv(datafile, header = TRUE, check.names = FALSE)
# read the model from the rds file

cardinalities <- data[1, ]

### search the cardinalities
data <- data[-1, ] ## remove first row

# convert to data frame
df <- as.data.frame(data)

start_time <- Sys.time()
model <- stagedtrees::search_best(df, alg = stages_bhc)
end_time <- Sys.time()

# write the model to the rds file
modelfile <- snakemake@output$st
saveRDS(model, modelfile)
# create a csv file with the time
timefile <- snakemake@output$time
total_time <- start_time - end_time

time <- data.frame(method = "bos", 
                   time = total_time,
                   total_time = total_time,
                   n=snakemake@wildcards$cstree_data_n,
                   seed=snakemake@wildcards$seed,
                   p=snakemake@wildcards$cstree_p)

write.csv(time, timefile, row.names = FALSE, quote = FALSE)
