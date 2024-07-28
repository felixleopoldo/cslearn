library(stagedtrees)
library(bnlearn)

matrixToDataframe <- function(data, varnames) {
  datadf <- data
  n <- dim(data)[2]
  colnames(datadf) <- varnames
  datadf <- as.data.frame(datadf)
  for (j in 1:ncol(datadf)) {
    datadf[, varnames[j]] <- as.factor(datadf[, varnames[j]])
  }
  return(datadf)
}

datafile <- snakemake@input$data

# read data from csv
data <- read.csv(datafile, header = TRUE, check.names = FALSE)
# read the model from the rds file

cardinalities <- data[1, ]

### search the cardinalities
data <- data[-1, ] ## remove first row

start_time <- Sys.time()

# read dag from file as csv
print("read dag")
dag_adjmat <- read.csv(snakemake@input$input_alg_dag, header = FALSE, check.names = FALSE)

dag_matrix <- as.matrix(dag_adjmat)

colnames(data) <- paste0("X",c(1:ncol(data)))

colnames(dag_matrix) <- colnames(data)
colnames(data) <- colnames(data)

print(colnames(data))

# convert to bnlearn dag
dag <- empty.graph(nodes = colnames(data))

# Set the adjacency matrix
amat(dag) <- dag_matrix

# Display the DAG
dag
print("fit")

#print(data)
# convert to data frame columns to factors
#df <- as.data.frame(data)
#df <- lapply(df, as.factor)
#print(df)


df <- matrixToDataframe(data, colnames(data))
#dagfit <- bnlearn::tabu(df)
dagfit <- bnlearn::bn.fit(dag, df)
print("fited dag")
print(dagfit)

model <- as_sevt(dagfit) |> sevt_fit(df, lambda = 0) |> stages_bhc()

#model <- stagedtrees::search_best(df, alg = stages_bhc)
end_time <- Sys.time()

# write the model to the rds file
modelfile <- snakemake@output$st
saveRDS(model, modelfile)
# create a csv file with the time
timefile <- snakemake@output$time
bhc_time <- start_time - end_time

# get the input algorithm time
print("read input time")
# read as ordinary file
input_alg_time <- readLines(snakemake@input$time)

#input_alg_time <- read.csv(snakemake@input$time, header = TRUE, check.names = FALSE)


print("input time")
print(input_alg_time$time)

total_time <- bhc_time + input_alg_time$time



print("total time")
print(total_time)
print("bhc time")
print(bhc_time)


time <- data.frame(method = "bhc", 
                   time = bhc_time,
                   total_time = total_time,
                   n=snakemake@wildcards$cstree_data_n,
                   seed=snakemake@wildcards$seed,
                   p=snakemake@wildcards$cstree_p)

write.csv(time, timefile, row.names = FALSE, quote = FALSE)
