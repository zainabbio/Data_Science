library(dslabs)
library(tidyverse)
library(caret)
library(corrplot)
library(matrixStats)



# Breast Cancer Wisconsin Diagnostic Dataset from UCI Machine Learning Repository
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

# Biopsy features for classification of 569 malignant (cancer) and benign (not cancer) breast masses.

# Features were computationally extracted from digital images of fine needle aspirate biopsy slides.
# Features correspond to properties of cell nuclei, such as size, shape and regularity. The mean,
# standard error, and worst value of each of 10 nuclear parameters is reported for a total of 30 features.

# script adapted from https://rpubs.com/leonardotamashiro/567038

data("brca")



# y. The outcomes. A factor with two levels denoting whether a mass 
# is malignant ("M") or benign ("B").

# x. The predictors (features). A matrix with the mean, standard error 
# and worst value of each of the 10 nuclear measurements on the slide, 
# for 30 total features per biopsy:
# – radius. Nucleus radius (mean of distances from center to points on perimeter).
# – texture. Nucleus texture (standard deviation of grayscale values).
# – perimeter. Nucleus perimeter.
# – area. Nucleus area.
# – smoothness. Nucleus smoothness (local variation in radius lengths).
# – compactness. Nucleus compactness (perimeter^2/area - 1).
# – concavity, Nucleus concavity (severity of concave portions of the contour).
# – concave_pts. Number of concave portions of the nucleus contour.
# – symmetry. Nucleus symmetry.
# – fractal_dim. Nucleus fractal dimension ("coastline approximation" -1).


dim(brca$x)
length(brca$y)



# How do we find how many samples are malignant and how many are benign?

table(brca$y)


# Find the proportion of B and M

prop.table(table(brca$y))


# transformation using scale() function
x_scl <- scale(brca$x)
colSds(x_scl)
summary(x_scl)



# Creating data partition (for training and testing)
# The function createDataPartition can be used to create balanced splits 
# of the data. If the y argument to this function is a factor, the random 
# sampling occurs within each class and should preserve the overall class 
# distribution of the data
set.seed(1,sample.kind = "Rounding")
test_index <- createDataPartition(brca$y, time=1, p=0.2,list=FALSE)
test_index

test_x <- x_scl[test_index,]

test_y <- brca$y[test_index]

train_x <- x_scl[-test_index,]

train_y <- brca$y[-test_index]

# Let us check whether we have similar split of outcome data in test and train
prop.table(table(test_y))
prop.table(table(train_y))





# What are the models available for train()?
names(getModelInfo())



# Now let us use Logistic regression as our first algorithm
# Logistic Regression

trained_glm <- train(train_x,train_y,method="glm")

# Check the 'trained_glm' model object for summary of methods
# What resampling method is used?
trained_glm



# Change the default resampling method of bootstrapping to 10-fold cross-validation
tc <- trainControl(method = "cv", number = 10)
trained_glm_cv <- train(train_x,train_y,method="glm", trControl = tc)


glm_preds <- predict(trained_glm,test_x)

attributes(trained_glm)

trained_glm$method
trained_glm$trainingData

# accuracy at each resampling step
trained_glm$resample

# shows resampling
trained_glm$control

# shows the coefficients from the final model
trained_glm$finalModel





#  finding the accuracy
confusionMatrix(glm_preds, test_y)$overall[["Accuracy"]]


# confusionMatrix can provide more measures of performance
confusionMatrix(glm_preds, test_y)
confusionMatrix(glm_preds, test_y, positive = "M")
confusionMatrix(glm_preds, test_y, positive = "M")$byClass










# Linear Discriminant Analysis (LDA) model
trained_lda <- train(train_x,train_y,method="lda")
lda_preds <- predict(trained_lda,test_x)
mean(lda_preds==test_y)





#  Quadratic Discriminant Analysis (QDA) model
trained_qda <- train(train_x,train_y,method="qda")
qda_preds <- predict(trained_qda,test_x)
mean(qda_preds==test_y)







### https://rpubs.com/uky994/593668

### Support Vector Machines (SVM)
# SVM methods can handle both linear and non-linear class boundaries. 
# It can be used for both two-class and multi-class classification problems. 
# In real life data, the separation boundary is generally nonlinear. 
# Technically, the SVM algorithm perform a non-linear classification using 
# what is called the kernel trick. The most commonly used kernel transformations
# are polynomial kernel and radial kernel.

# First let us run linear SVM

svm1 <- train(train_x, train_y, method = "svmLinear")

train_control <- trainControl(method="boot", number=100)
svm1 <- train(train_x, train_y, method = "svmLinear", trControl = train_control)




# Set up Repeated k-fold Cross Validation
train_control <- trainControl(method="repeatedcv", number=10, repeats=3)
svm1 <- train(train_x, train_y, method = "svmLinear", trControl = train_control)
svm1



#  Tuning parameter C (cost) determines the possible misclassifications. 
# It  imposes a penalty to the model for making an error: 
# the higher the value of C, the less likely it is that the SVM algorithm 
# will misclassify a point.
#
#It’s possible to automatically compute SVM for different values of C and 
# to choose the optimal one that maximize the model cross-validation accuracy.

# Let us change default of C = 1 for a grid values of C  to choose automatically
# the final model for predictions

svm2 <- train(train_x, train_y, method = "svmLinear", trControl = train_control,
              tuneGrid = expand.grid(C = seq(0, 2, length = 20)))
#View the model
svm2

# Plot model accuracy vs different values of Cost
plot(svm2)

# Print the best tuning parameter C that maximizes model accuracy
svm2$bestTune

# save the results
res2<-as_tibble(svm2$results[which.max(svm2$results[,2]),])
res2






## SVM classifier using Non-Linear Kernel
# To build a non-linear SVM classifier, we can use polynomial kernel 
# or radial basis kernel function.


# Computing SVM using radial basis kernel
svm3 <- train(train_x, train_y, method = "svmRadial", trControl = train_control,
              tuneLength = 10)
# Print the best tuning parameter sigma and C that maximizes model accuracy
svm3
svm3$bestTune
# Plot model accuracy vs different values of Cost
plot(svm3)

#save the results for later
res3<-as_tibble(svm3$results[which.max(svm3$results[,3]),])
res3





## Computing SVM using polynomial basis kernel:
# Fit the model 
svm4 <- train(train_x, train_y, method = "svmPoly", trControl = train_control, 
              tuneLength = 4)




svm4 <- train(train_x, train_y, method = "svmPoly", 
              tuneGrid = expand.grid(degree = 1, scale = FALSE, C = seq(0, 2, length = 20)))

# Print the best tuning parameter sigma and C that maximizes model accuracy
svm4$bestTune
svm4
plot(svm4)

#save the results for later
res4<-as_tibble(svm4$results[which.min(svm4$results[,2]),])
res4

## compare the results
df<-tibble(Model=c('SVM Linear','SVM Linear w/ choice of cost','SVM Radial','SVM Poly'),
           Accuracy=c(svm1$results[2][[1]],res2$Accuracy,res3$Accuracy,res4$Accuracy))
df %>% arrange(Accuracy)



### Classify the held-out test samples
svm1_preds <- predict(svm1,test_x)
confusionMatrix(svm1_preds, test_y)$overall[["Accuracy"]]

svm2_preds <- predict(svm2,test_x)
confusionMatrix(svm2_preds, test_y)$overall[["Accuracy"]]

svm3_preds <- predict(svm3,test_x)
confusionMatrix(svm3_preds, test_y)$overall[["Accuracy"]]

svm4_preds <- predict(svm4,test_x)
confusionMatrix(svm4_preds, test_y)$overall[["Accuracy"]]



#########
### Classification and regression trees (CART)
# use rpart model (Recursive Partitioning and Regression Trees)
train_control <- trainControl(method="repeatedcv", number=10, repeats=3)

cart1 <- train(train_x, train_y, method = "rpart", trControl = train_control)
cart1
# The complexity parameter (cp) in rpart is the minimum improvement in the model
# needed at each node. 
cart1_preds <- predict(cart1, test_x)
confusionMatrix(cart1_preds, test_y)$overall[["Accuracy"]]





# Random Forest Model
set.seed(9, sample.kind = "Rounding")
# set the number of variables randomly sampled as candidates at each split in
# random forests 
tuning <- data.frame(mtry=c(3,5,7,9))
trained_rf <- train(train_x,train_y, method="rf",
                    tuneGrid = tuning, importance = TRUE)
trained_rf$bestTune

trellis.par.set(caretTheme())
plot(trained_rf)


rf_preds <- predict(trained_rf, test_x)
mean(rf_preds == test_y)





# K-nearest neighbors
# With default parameters
set.seed(7, sample.kind = "Rounding")
trained_knn <- train(train_x,train_y,method="knn")
ggplot(trained_knn, highlight = T)

# Specify explicitly the number of bootstrapping for resampling
tc <- trainControl(method = "boot", number = 100)
trained_knn <- train(train_x,train_y,method="knn", trControl = tc)
# From the plot, what are the default values of K (#neighbors)?






# Set the number of neighbors 
set.seed(7, sample.kind = "Rounding")
tuning <- data.frame(k=seq(3,21,2))
trained_knn <- train(train_x,train_y,method="knn",tuneGrid = tuning)
ggplot(trained_knn, highlight = T)

# Find the parameter that maximized the accuracy of the model
trained_knn$bestTune

# Which is the best performing model
trained_knn$finalModel





# Change the default resampling method of bootstrapping to 10-fold cross-validation
tc <- trainControl(method = "cv", number = 10)
trained_knn_cv <- train(train_x,train_y,method="knn", trControl = tc)
ggplot(trained_knn_cv, highlight = T)






# Predict the held-out test dataset with the trained model
knn_preds <- predict(trained_knn,test_x)

# What is the performance of the model on the test data?
mean(knn_preds == test_y)
confusionMatrix(knn_preds, test_y)$overall[["Accuracy"]]
confusionMatrix(knn_preds, test_y, positive = "M")











# Random Forest Model
set.seed(9, sample.kind = "Rounding")
tuning <- data.frame(mtry=c(3,5,7,9))
trained_rf <- train(train_x,train_y, method="rf",tuneGrid = tuning, importance = TRUE)
trained_rf$bestTune

rf_preds <- predict(trained_rf, test_x)
mean(rf_preds == test_y)







#### Neural networks

numFolds <- trainControl(method = 'cv', number = 10, 
                         verboseIter = TRUE)
nnet1 <- train(train_x, train_y, method = 'nnet',
               trControl = numFolds, tuneGrid=expand.grid(size=seq(10,20, 1), decay=c(0.1)))

nnet1_preds <- predict(nnet1, test_x)
confusionMatrix(nnet1_preds, test_y)$overall[["Accuracy"]]

confusionMatrix(nnet1_preds, test_y)





### Model comparison ####
## Using compare_models()
compare_models(svm1_preds, cart1_preds)






## Important Features 

svm1_imp_features <- varImp(svm1)
plot(svm1_imp_features)

cart1_imp_features <- varImp(cart1)
plot(cart1_imp_features)


nnet1_imp_features <- varImp(nnet1)
plot(nnet1_imp_features)




# K-means Clustering
predict_kmeans <- function(x, k) {
  centers <- k$centers 
  distances <- sapply(1:nrow(x), function(i){
    apply(centers, 1, function(y) dist(rbind(x[i,], y)))
  })
  max.col(-t(distances)) 
}

set.seed(3,sample.kind = "Rounding")
k <- kmeans(train_x, centers = 2)
kmeans_preds <- ifelse(predict_kmeans(test_x, k) == 1, "B", "M")

# K-means overall accuracy
mean(kmeans_preds == test_y)

table(test_y,kmeans_preds)

sensitivity(factor(kmeans_preds), test_y, positive = "B")
sensitivity(factor(kmeans_preds), test_y, positive = "M")



# Hierarchical clustering
h <- hclust(d_features)



# Split the tree and see which features are in the same cluster
groups <- cutree(h,k=5)
split(names(groups),groups)



# plot the dendrogram, with 
# smaller labels to fit (cex 50%), and labels at same height (hang -1)
plot(h, cex = 0.5, hang = -1)

# or better use the ggdendro package
library(ggdendro)
ggdendrogram(h)
ggdendrogram(h, rotate = T)







# Dimension reduction: PCA
pca <- prcomp(x_scaled)
summary(pca)


# Plotting PCs, we can see the benign tumors tend to have smaller values of PC1 and 
# higher values for malignant tumors
data.frame(pca$x[,1:2],type = brca$y) %>%
  ggplot(aes(PC1,PC2,color=type)) + 
  geom_point()


# scatterplot of PC2 versus PC1 with an ellipse to show the cluster regions
data.frame(pca$x[,1:2], type = ifelse(brca$y == "B", "Benign", "Malignant")) %>%
  ggplot(aes(PC1, PC2, color = type)) +
  geom_point() +
  stat_ellipse() +
  ggtitle("PCA separates breast biospies into benign and malignant clusters")




# Plotting first 10 PCs as boxplot.  We can see PC1 is significantly different from others
data.frame(type = brca$y ,pca$x[,1:10]) %>%
  gather(key = "PC",value="value", -type) %>%
  ggplot(aes(PC,value,fill = type)) +
  geom_boxplot()


#geom_point()





























