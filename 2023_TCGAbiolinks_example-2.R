###
## TCGABiolinks
## To query, prepare and download data from the TCGA portal
##
## TCGAbiolinks provides important functionality as matching data of same
## donors across distinct data types (clinical vs expression) and 
## provides data structures to make its analysis in R easy

# https://costalab.ukaachen.de/open_data/Bioinformatics_Analysis_in_R_2019/BIAR_D3/handout.html

## To download TCGA data with TCGAbiolinks, you need to follow 3 steps. 
## First, you will query the TCGA database through R with the function GDCquery.
## This will allow you to investigate the data available at the TCGA database. 
## Next, we use GDCdownload to download raw version of desired files into 
## your computer. 
## Finally GDCprepare will read these files and make R data structures 
## so that we can further analyse them.



if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("SummarizedExperiment")
BiocManager::install("TCGAbiolinksGUI.data")
BiocManager::install("BioinformaticsFMRP/TCGAbiolinks")
BiocManager::install("edgeR")


library("TCGAbiolinks", quietly = T)
library("SummarizedExperiment", quietly = T)
library("edgeR")
library("caret", quietly = T)
library(survival)
library(dplyr)
library(plyr)
library(tidyr)
library(tidyverse)
library(caret)
library(corrplot)
library(matrixStats)
library(gprofiler2)
library(limma)
library(survminer)
library(edgeR)
library(SummarizedExperiment)
library(genefilter)
library(glmnet)
library(factoextra)
library(FactoMineR)
library(gplots)
library(RColorBrewer)

### 
## First, check all the available projects at TCGA 
getGDCprojects()
GDCprojects = getGDCprojects()

head(GDCprojects[c("project_id", "name")])

View(GDCprojects[c("project_id", "name")])


## Example dataset: Stomach Adenocarcinoma, identified in TCGA as KIRC
# get details on all data deposited for TCGA-KIRC.
TCGAbiolinks:::getProjectSummary("TCGA-READ")


## querying all Transcriptome profiling data from KIRC project
read_TCGA = GDCquery(
  project = "TCGA-READ",
  data.category = "Transcriptome Profiling")


# Visualize the results
View(getResults(read_TCGA))


## querying all RNA-seq data from KIRC project
read_TCGA = GDCquery(
  project = "TCGA-READ",
  data.category = "Transcriptome Profiling",
  data.type = "Gene Expression Quantification",
  experimental.strategy = "RNA-Seq",
  workflow.type = "STAR - Counts")


# visualize the query results in a more readable way
read_res = getResults(read_TCGA) # make results as table
head(read_res) # data of the first 6 patients.
View(getResults(read_TCGA))
colnames(read_res) # columns present in the table


# Now get tissue type measured at an experiment (normal, solid tissue, cell line).
# This information is present at column “sample_type”.

print(read_res$sample_type)
# convert sample_type to factor and summarise using summary or table
table(factor(read_res$sample_type))

# there are 50 controls (Solid Tissue Normal) and 371 cancer samples (Primary Tumors). 
# For simplicity, we will ignore the small class of recurrent solid tumors. 
# Therefore, we will redo the query
read_TCGA = GDCquery(
  project = "TCGA-READ",
  data.category = "Transcriptome Profiling", # parameter enforced by GDCquery
  experimental.strategy = "RNA-Seq",
  workflow.type = "STAR - Counts",
  sample.type = c("Primary Tumor", "Solid Tissue Normal"),
  data.type = "Gene Expression Quantification")

# visualize the query results in a more readable way
read_res = getResults(read_TCGA) # make results as table
head(read_res) # data of the first 6 patients.
# convert sample_type to factor and summarise the sample types using summary() or table()
table(factor(read_res$sample_type)) 


### Download the files from the query
# first set the directory to download
getwd()
setwd("~/Downloads/Final_project/")

# download the files specified in the query.
GDCdownload(query = read_TCGA) # rerun if download fails


### Prepare the downloaded data and other associated data into an object
## load the actual RNASeq data into R
## both clinical and expression data will be present in this object
read_data = GDCprepare(read_TCGA)
View(read_data)

dim(read_data)
str(read_data)
?SummarizedExperiment


colnames(colData(read_data))

table(read_data@colData$vital_status)
table(read_data@colData$tumor_grade)
table(read_data@colData$definition)
table(read_data@colData$tissue_or_organ_of_origin)
table(read_data@colData$gender)
table(read_data@colData$race)
table(read_data@colData$sample_type)
table(read_data@colData$ajcc_pathologic_stage)

# RNAseq data
dim(assay(read_data))     # gene expression matrices.
head(assay(read_data)[,1:10]) # expression of first 6 genes and first 10 samples
head(rowData(read_data))     # ensembl id and gene id of the first 6 genes.


# Save the data as a file, if you need it later, you can just load this file
# instead of having to run the whole pipeline again
saveRDS(object = read_data,
        file = "read_data.RDS",
        compress = FALSE)

## Later it can be retrieved
# read_data = readRDS(file = "read_data.RDS")
#Survival Analysis
read_clinical = read_data@colData
colnames(read_clinical)
table(factor(read_clinical$ajcc_pathologic_stage)) 

read_df = read_clinical[c("definition",
                     "patient",
                     "vital_status",
                     "days_to_death",
                     "days_to_last_follow_up",
                     "ajcc_pathologic_stage")]

read_df$deceased = read_df$vital_status == "Dead"

read_df$overall_survival = ifelse(read_df$deceased,
                                  read_df$days_to_death,
                                  read_df$days_to_last_follow_up)

#To generate an overall survival curve
fit_overall = survfit(Surv(overall_survival, deceased) ~ 1, data=read_df)
ggsurvplot(fit_overall, data=read_df, pval=T, risk.table=T, risk.table.col="strata", palette = c("#2E9FDF"))


#To generate tumor-stage specific survival surve:
read_df$tumor_stage = gsub("[abc]$", "", read_clinical$ajcc_pathologic_stage)
read_df[which(read_clinical$ajcc_pathologic_stage == "not reported"), "ajcc_pathologic_stage"] = NA

table(read_df$ajcc_pathologic_stage)

fit_pathological_stage = survfit(Surv(overall_survival, deceased) ~ ajcc_pathologic_stage, data=read_df)
ggsurvplot(fit_pathological_stage, data=read_df, pval=T, risk.table=F, surv.plot.height=0.7, legend.labs=c("Stage I", "Stage II", "Stage IIA", "Stage IIB", "Stage IIC","Stage III", "Stage IIIA", "Stage IIIB", "Stage IIIC","Stage IV", "Stage IVA"))



##### RNAseq analysis #########
## create a pipeline function for running limma
## The function is called limma_pipeline(tcga_data, condition_variable, reference_group), 
# where tcga_data is the data object we got from TCGA and 
# condition_variable is the interesting variable/condition by which 
# we want to group the patient samples. 
# You can also decide which one of the values of your conditional variable 
# is going to be the reference group, with the reference_group parameter.

limma_pipeline = function(
  tcga_data,
  condition_variable,
  reference_group=NULL){
  
  design_factor = colData(tcga_data)[, condition_variable, drop=T]
  
  group = factor(design_factor)
  if(!is.null(reference_group)){group = relevel(group, ref=reference_group)}
  
  design = model.matrix(~ group)
  
  dge = DGEList(counts=assay(tcga_data),
                samples=colData(tcga_data),
                genes=as.data.frame(rowData(tcga_data)))
  
  # filtering
  keep = filterByExpr(dge,design)
  dge = dge[keep,,keep.lib.sizes=FALSE]
  rm(keep)
  
  # Normalization (TMM followed by voom)
  dge = calcNormFactors(dge)
  v = voom(dge, design, plot=TRUE)
  
  # Fit model to data given design
  fit = lmFit(v, design)
  fit = eBayes(fit)
  
  # Show top genes
  topGenes = topTable(fit, coef=ncol(design), number=100, sort.by="p")
  
  return(
    list(
      voomObj=v, # normalized data
      fit=fit, # fitted model and statistics
      topGenes=topGenes # the 100 most differentially expressed genes
    )
  )
}

############
##
# This function returns a list with three different objects:
  
# A complex object, resulting from running voom, this contains the TMM+voom normalized data;
# A complex object, resulting from running eBayes, this contains the the fitted 
# model plus a number of statistics related to each of the probes;
# A simple table, resulting from running topTable, which contains the top 100 
# differentially expressed genes sorted by p-value.

################################






## Differential expression analysis
# DE analysis comparing Primary solid Tumor samples against Solid Tissue Normal
limma_res = limma_pipeline(
  tcga_data=tcga_data,
  condition_variable="definition",
  reference_group="Solid Tissue Normal"
)

View(limma_res)


# Save the data as a file, if you need it later, you can just load this file
# instead of having to run the whole pipeline again
saveRDS(object = limma_res,
        file = "limma_res.RDS",
        compress = FALSE)





##### Classification ####
#  start by extracting the data that we are going to use to build our model.
# We want the expression data that has already been normalized and 
# a clinical feature which divides our data into different groups, 
# such as tumor vs. non-tumor or tumor stage. 
# We can get the normalized expression values from limma_res$voomObj$E 
# and the type of sample is determined by the definition column.

# Transpose and make it into a matrix object
d_mat = as.matrix(t(limma_res$voomObj$E))

# Convert the definition into a factor
d_resp = as.factor(limma_res$voomObj$targets$definition)
table(d_resp)
### Split the data into train and test sets
# Divide data into training and testing set

# Set (random-number-generator) seed so that results are consistent between runs
set.seed(42)
train_ids = createDataPartition(d_resp, p=0.75, list=FALSE)

x_train = d_mat[train_ids, ]
x_test  = d_mat[-train_ids, ]

y_train = d_resp[train_ids]
y_test  = d_resp[-train_ids]




### Train an elastic net model
# which is a generalized linear model that combines the best of two other models:
# LASSO and Ridge Regression.

# Train model on training dataset using cross-validation
res = cv.glmnet(
  x = x_train,
  y = y_train,
  alpha = 0.5,
  family = "binomial"
)




## Model evaluation
## validate using test set
y_pred = predict(res, newx=x_test, type="class", s="lambda.min")


# performance
confusion_matrix = table(y_pred, y_test)

# Evaluation statistics
print(confusion_matrix)
print(paste0("Sensitivity: ",sensitivity(confusion_matrix)))
print(paste0("Specificity: ",specificity(confusion_matrix)))
print(paste0("Precision: ",precision(confusion_matrix)))




# Getting genes that contribute for the prediction
res_coef = coef(res, s="lambda.min") # the "coef" function returns a sparse matrix
dim(res_coef)

head(res_coef) # in a sparse matrix the "." represents the value of zero

# get coefficients with non-zero values
res_coef = res_coef[res_coef[,1] != 0,]
# note how performing this operation changed the type of the variable
head(res_coef)


# remove first coefficient as this is the intercept, a variable of the model itself
res_coef = res_coef[-1]

relevant_genes = names(res_coef) # get names of the (non-zero) variables.
length(relevant_genes) # number of selected genes

head(relevant_genes) # few select genes

# get the common gene name from limma_res$voomObj$genes
head(limma_res$voomObj$genes)

relevant_gene_names = limma_res$voomObj$genes[relevant_genes,"external_gene_name"]

head(relevant_gene_names) # few select genes (with readable names now)

# Did limma and Elastic Net select some of the same genes?
print(intersect(limma_res$topGenes$gene_id, relevant_genes))



#### Hierarchical clustering ######
# look at the genes Elastic Net found to cluster the samples

# define the color palette for the plot
hmcol = colorRampPalette(rev(brewer.pal(9, "RdBu")))(256)


# perform complete linkage clustering
clust = function(x) hclust(x, method="complete")
# use the inverse of correlation as distance.
dist = function(x) as.dist((1-cor(t(x)))/2)

# Show green color for genes that also show up in DE analysis
colorLimmaGenes = ifelse(
  # Given a vector of boolean values
  (relevant_genes %in% limma_res$topGenes$ensembl_gene_id),
  "green", # if true, return green for that value
  "white" # if false, return white for that value
)

# heatmap involves a lot of parameters
gene_heatmap = heatmap.2(
  t(d_mat[,relevant_genes]),
  scale="row",          # scale the values for each gene (row)
  density.info="none",  # turns off density plot inside color legend
  trace="none",         # turns off trace lines inside the heat map
  col=hmcol,            # define the color map
  labRow=relevant_gene_names, # use gene names instead of ensembl annotation
  RowSideColors=colorLimmaGenes,
  labCol=FALSE,         # Not showing column labels
  ColSideColors=as.character(as.numeric(d_resp)), # Show colors for each response class
  dendrogram="both",    # Show dendrograms for both axis
  hclust = clust,       # Define hierarchical clustering method
  distfun = dist,       # Using correlation coefficient for distance function
  cexRow=.6,            # Resize row labels
  margins=c(1,5)        # Define margin spaces
)



#####
# get the dendrogram from the heatmap
# and cut it to get the 2 classes of genes

# Extract the hierarchical cluster from heatmap to class "hclust"
hc = as.hclust(gene_heatmap$rowDendrogram)

# Cut the tree into 2 groups, up-regulated in tumor and up-regulated in control
clusters = cutree(hc, k=2)
table(clusters)


# selecting just a few columns so that its easier to visualize the table
gprofiler_cols = c("significant","p.value","overlap.size","term.id","term.name")

# make sure the URL uses https
set_base_url("https://biit.cs.ut.ee/gprofiler")

# Group 1, up in tumor
gprofiler(names(clusters[clusters %in% 1]))[, gprofiler_cols]


# Group 2, up in control
gprofiler(names(clusters[clusters %in% 2]))[, gprofiler_cols]


############


# Now let us use Logistic regression as our first algorithm
# Logistic Regression

trained_glm <- train(x_train_coad,y_train_coad,method="glm")

# Check the 'trained_glm' model object for summary of methods
# What resampling method is used?
trained_glm



# Change the default resampling method of bootstrapping to 10-fold cross-validation
tc <- trainControl(method = "cv", number = 10)
trained_glm_cv <- train(x_train_coad,y_train_coad,method="glm", trControl = tc)


glm_preds <- predict(trained_glm,x_test_coad)

attributes(trained_glm)


trained_glm$trainingData

# accuracy at each resampling step
trained_glm$resample

# shows resampling
trained_glm$control

# shows the coefficients from the final model
trained_glm$finalModel





# Another way of finding the accuracy
confusionMatrix(glm_preds, test_y)$overall[["Accuracy"]]


# confusionMatrix can provide more measures of performance
confusionMatrix(glm_preds, test_y)
confusionMatrix(glm_preds, test_y, positive = "M")
confusionMatrix(glm_preds, test_y, positive = "M")$byClass




###########



##########
### Use SVM and train from caret
trained_svm <- train(x_train,y_train,method="svmLinear")
trained_svm$bestTune
svm_preds <- predict(trained_svm,x_test)

confusionMatrix(svm_preds, y_test)



