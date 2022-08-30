# Assignment 2

# install and import the required packages
install.packages("tree")
library(tree)
install.packages("e1071")
library(e1071)
install.packages(("ROCR"))
library(ROCR)
install.packages("randomForest")
library(randomForest)
install.packages("adabag")
library(adabag)
install.packages("rpart")
library(rpart)
install.packages("car")
library(car)
library(dplyr)

detach("package:neuralnet", unload = TRUE)

# creating the data set 
rm(list = ls())
WAUS <- read.csv("WarmerTomorrow2022.csv")
L <- as.data.frame(c(1:49))
set.seed(30899559) # Your Student ID is the random seed
L <- L[sample(nrow(L), 10, replace = FALSE),] # sample 10 locations
WAUS <- WAUS[(WAUS$Location %in% L),]
WAUS <- WAUS[sample(nrow(WAUS), 2000, replace = FALSE),] # sample 2000 rows


# ----------------------- Question 1 -------------------------

# get the number of days when it is warmer than the previous day
warmerCount <- nrow(WAUS[WAUS$WarmerTomorrow == 1,])
# proportion of days when it is warmer than the previous day
propWarmer <- warmerCount/nrow(WAUS)
# get the number of days when it is colder than the previous day
colderCount <- nrow(WAUS[WAUS$WarmerTomorrow == 0,])
# proportion of days when it is colder than the previous day
propColder <- colderCount/nrow(WAUS)

# see the results
propWarmer
propColder

# description of the predictors
summary(WAUS)


# ----------------------- Question 2 -------------------------

# make the attributes as factor
WAUS <- as.data.frame(unclass(WAUS), stringsAsFactors = TRUE)
WAUS$WarmerTomorrow <- factor(WAUS$WarmerTomorrow)

# remove NA values from the data 
WAUS <- WAUS[complete.cases(WAUS),]


# ----------------------- Question 3 -------------------------

# partition data into training and testing data 
set.seed(30899559) #Student ID as random seed
train.row = sample(1:nrow(WAUS), 0.7*nrow(WAUS))
WAUS.train = WAUS[train.row,]
WAUS.test = WAUS[-train.row,]

# ----------------------- Question 4 -------------------------

# fit decision tree model 
set.seed(30899559)
WAUS.tree <- tree(WarmerTomorrow ~ . , data = WAUS.train) 
plot(WAUS.tree)
text(WAUS.tree)

#fit Naive Bayes model 
set.seed(30899559)
WAUS.nb <- naiveBayes(WarmerTomorrow ~ . , data = WAUS.train) 

# fit a bagging model 
set.seed(30899559)
WAUS.bag <- bagging(WarmerTomorrow ~ . , data = WAUS.train, mfinal = 5) 

# fit a boosting model 
set.seed(30899559)
WAUS.boost <- boosting(WarmerTomorrow ~ . , data = WAUS.train, mfinal = 10) 

# fit a random forest model 
set.seed(30899559)
WAUS.rf <- randomForest(WarmerTomorrow ~ ., data = WAUS.train)



# ----------------------- Question 5 -------------------------

# make the prediction using decision tree model 
WAUS.tree.pred <- predict(WAUS.tree, WAUS.test, type = "class")
# create confusion matrix and calculate accuracy
WAUS.tree.cf <- table(predicted = WAUS.tree.pred, actual = WAUS.test$WarmerTomorrow)
WAUS.tree.cf
WAUS.tree.acc <- (WAUS.tree.cf[1,1] + WAUS.tree.cf[2,2]) /sum(WAUS.tree.cf)

# make the prediction for Naive Bayes model 
WAUS.nb.pred <- predict(WAUS.nb, WAUS.test, type = "raw")
WAUS.nb.cpred <- predict(WAUS.nb, WAUS.test, type = "class")
# create confusion matrix and calculate accuracy
WAUS.nb.cf <-table(predicted = WAUS.nb.cpred, actual = WAUS.test$WarmerTomorrow)
WAUS.nb.cf
WAUS.nb.acc <- (WAUS.nb.cf[1,1] + WAUS.nb.cf[2,2]) /sum(WAUS.nb.cf)

# make the prediction for bagging model
WAUS.bag.pred <- predict(WAUS.bag, WAUS.test, type = "raw")
# create confusion matrix and calculate accuracy
WAUS.bag.cf <-WAUS.bag.pred$confusion
WAUS.bag.cf
WAUS.bag.acc <- (WAUS.bag.cf[1,1] + WAUS.bag.cf[2,2]) /sum(WAUS.bag.cf)

# make the prediction for boosting model 
WAUS.boost.pred <- predict(WAUS.boost, WAUS.test, type = "raw")
# create confusion matrix and calculate accuracy
WAUS.boost.cf <- WAUS.boost.pred$confusion
WAUS.boost.cf
WAUS.boost.acc <- (WAUS.boost.cf[1,1] + WAUS.boost.cf[2,2]) /sum(WAUS.boost.cf)

# make the prediction for random forest model
WAUS.rf.pred <- predict(WAUS.rf, WAUS.test)
# create confusion matrix and calculate accuracy
WAUS.rf.cf <-table(predicted = WAUS.rf.pred, actual = WAUS.test$WarmerTomorrow)
WAUS.rf.cf
WAUS.rf.acc <- (WAUS.rf.cf[1,1] + WAUS.rf.cf[2,2]) /sum(WAUS.rf.cf)


# ----------------------- Question 6 -------------------------

# ROC curve for decision tree model 
WAUS.tree.pred.vec <- predict(WAUS.tree, WAUS.test, type = "vector")
WAUSdPred <- prediction(WAUS.tree.pred.vec[,2], WAUS.test$WarmerTomorrow)
WAUSdPerf <- performance(WAUSdPred, "tpr", "fpr")
plot(WAUSdPerf)
abline(0,1)
# calculate the AUC
WAUS.tree.auc <- as.numeric(performance(WAUSdPred,"auc")@y.values)

# ROC curve for Naive Bayes model
WAUSdPred <- prediction(WAUS.nb.pred[,2], WAUS.test$WarmerTomorrow)
WAUSdPerf <- performance(WAUSdPred, "tpr", "fpr")
plot(WAUSdPerf, add = TRUE, col = "green")
# calculate the AUC
WAUS.nb.auc <- as.numeric(performance(WAUSdPred,"auc")@y.values)

# ROC curve for bagging model 
WAUSdPred <- prediction(WAUS.bag.pred$prob[,2], WAUS.test$WarmerTomorrow)
WAUSdPerf <- performance(WAUSdPred, "tpr", "fpr")
plot(WAUSdPerf, add = TRUE, col = "red")
# calculate the AUC
WAUS.bag.auc <- as.numeric(performance(WAUSdPred,"auc")@y.values)

# ROC curve for boosting model 
WAUSdPred <- prediction(WAUS.boost.pred$prob[,2], WAUS.test$WarmerTomorrow)
WAUSdPerf <- performance(WAUSdPred, "tpr", "fpr")
plot(WAUSdPerf, add = TRUE, col = "blue")
# calculate the AUC
WAUS.boost.auc <- as.numeric(performance(WAUSdPred,"auc")@y.values)

# ROC curve for random forest model 
WAUSpred.rf <- predict(WAUS.rf, WAUS.test, type = "prob")
WAUSdPred <- prediction(WAUSpred.rf[,2], WAUS.test$WarmerTomorrow)
WAUSdPerf <- performance(WAUSdPred, "tpr", "fpr")
plot(WAUSdPerf, add = TRUE, col = "brown")
# calculate the AUC
WAUS.rf.auc <- as.numeric(performance(WAUSdPred,"auc")@y.values)


# ----------------------- Question 7 -------------------------

# create a table to collect the accutacy and area under the curve from all models and combine it
model <- c("Decision tree", "Naive Bayes", "Bagging", "Boosting", "Random Forest")
accuracy <- c(WAUS.tree.acc, WAUS.nb.acc, WAUS.bag.acc, WAUS.boost.acc, WAUS.rf.acc)
auc <- c(WAUS.tree.auc, WAUS.nb.auc, WAUS.bag.auc, WAUS.boost.auc, WAUS.rf.auc)
results <- data.frame(accuracy, auc)
rownames(results) <- model


# ----------------------- Question 8 -------------------------

# get variable importance, in decision treel the most important is WindDir9am
print(summary(WAUS.tree))

# get variable importance, in bagging the most important is WindDir9am omit rainfall
print(WAUS.bag$importance)

# get variable importance, in boosting the most important is WindGustDir  and omit rainfall
print(WAUS.boost$importance)

# get variable importance, in random forest the most important is WindDir9am and omit Location
print(WAUS.rf$importance)
varImpPlot(WAUS.rf)


# ----------------------- Question 9 -------------------------

# perfrom cross validation test 
cvtest<- cv.tree(WAUS.tree, FUN = prune.misclass)
cvtest

# prune using size 2 for simplicity
prune.zfit <- prune.misclass(WAUS.tree, best=2)
print(summary(prune.zfit))
plot(prune.zfit)
text(prune.zfit)

# do prediction and get make confusion matrix to get the accuracy
prune.zfit.pred <- predict(prune.zfit, WAUS.test, type = "class")
prune.zfit.cf <- table(predicted = prune.zfit.pred, actual = WAUS.test$WarmerTomorrow)
prune.zfit.acc <- (prune.zfit.cf[1,1] + prune.zfit.cf[2,2]) /sum(prune.zfit.cf)
prune.zfit.acc

# calculate the ROC curve 
WAUS.pruned.tree.pred.vec <- predict(prune.zfit, WAUS.test, type = "vector")
WAUSdPred <- prediction(WAUS.pruned.tree.pred.vec[,2], WAUS.test$WarmerTomorrow)
WAUSdPerf <- performance(WAUSdPred, "tpr", "fpr")
# calculate the AUC
WAUS.pruned.tree.auc <- as.numeric(performance(WAUSdPred,"auc")@y.values)
WAUS.pruned.tree.auc


# ----------------------- Question 10  -------------------------

# read the data again as in the first part of the script
w <- read.csv("WarmerTomorrow2022.csv")
L <- as.data.frame(c(1:49))
set.seed(30899559) # Your Student ID is the random seed
L <- L[sample(nrow(L), 10, replace = FALSE),] # sample 10 locations
w <- w[(w$Location %in% L),]
w <- w[sample(nrow(w), 2000, replace = FALSE),] # sample 2000 rows

# remove the Cloud3pm and Cloud 9am column 
w <- subset(w, select= -c(Cloud3pm, Cloud9am))
# make the attributes as factor
w <- as.data.frame(unclass(w), stringsAsFactors = TRUE)
w$WarmerTomorrow <- factor(w$WarmerTomorrow)
# remove NA values from the data 
w <- w[complete.cases(w),]

# partition data into training and testing data 
set.seed(30899559) #Student ID as random seed
train.row = sample(1:nrow(w), 0.7*nrow(w))
w.train = w[train.row,]
w.test = w[-train.row,]

# fit an improved random forest model 
set.seed(30899559)
w.rf <- randomForest(WarmerTomorrow ~ ., data = w.train, ntree = 500)

# make the prediction for the impoved random forest model
w.rf.pred <- predict(w.rf, w.test)
# create confusion matrix and calculate accuracy
w.rf.cf <-table(predicted = w.rf.pred, actual = w.test$WarmerTomorrow)
w.rf.cf
w.rf.acc <- (w.rf.cf[1,1] + w.rf.cf[2,2]) /sum(w.rf.cf)
w.rf.acc

# ROC curve for improved random forest model 
wpred.rf <- predict(w.rf, w.test, type = "prob")
wPred <- prediction(wpred.rf[,2], w.test$WarmerTomorrow)
wPerf <- performance(wPred, "tpr", "fpr")
# calculate the AUC
w.rf.auc <- as.numeric(performance(wPred,"auc")@y.values)
w.rf.auc

# ----------------------- Question 11  -------------------------

library(neuralnet)

# get the data and turn the target variable as numeric 
N <- WAUS
N$WarmerTomorrow <- as.numeric(N$WarmerTomorrow)

# split into training and testing dataset 
train.row = sample(1:nrow(N), 0.8*nrow(N)) 
N.train = N[train.row,]
N.test = N[-train.row,]

# craete the neural network model with selected attributes
set.seed(30899559)
N.net <- neuralnet(WarmerTomorrow == 1 ~ Sunshine +  MaxTemp + Humidity9am + Humidity3pm, N.train, hidden=3,linear.output = FALSE)

# do the test and get the result of the model
N.pred = compute(N.net, N.test[c("Sunshine", "MaxTemp", "Humidity9am", "Humidity3pm")])
prob <- N.pred$net.result
pred <- ifelse(prob>0.5, 1, 0)
# create confusion matrix
N.cf <- table(observed = N.test$WarmerTomorrow, predicted = pred)
N.cf
# compute accuracy 
N.cf.acc <- (N.cf[1,1] + N.cf[2,2]) /sum(N.cf)
N.cf.acc


