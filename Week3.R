# Course8 Practical Machine Learning

# Week3
data("iris"); library(ggplot2)
names(iris)
table(iris$Species)

inTrain <- createDataPartition(y = iris$Species, p = 0.7, list = FALSE)
training <- iris[inTrain, ]
testing <- iris[-inTrain, ]
dim(training); dim(testing)

qplot(Petal.Width, Sepal.Width, color = Species, data = training) #3 distinct clusters

library(caret)
modFit <- train(Species ~ ., method = "rpart", data = training)
#rpart: regression and classification
print(modFit$finalModel)

# plot tree
plot(modFit$finalModel, uniform = TRUE, main = "Classification Tree")
text(modFit$finalModel, use.n = TRUE, all = TRUE, cex = 0.8)

# prettier plots
library(rattle)
fancyRpartPlot(modFit$finalModel)

# predict new values
predict(modFit, newdata = testing)

# classification trees are non-linear models
# use interactions, can overfit if there are too many variables
# data transformation may be less important (monotone)
# RMSQ can be used for impurity

################################################################################
# bagging (bootstrap aggregating)
# average models together to get a smoother model fit (better balance between bias and variance)
# basic idea: resample cases and recalculate predictions
# average the predictions or majority vote
# similar bias as any individual model but reduce variance
# most useful for non-linear models

library(ElemStatLearn); data("ozone", package = "ElemStatLearn")
ozone <- ozone[order(ozone$ozone), ]
head(ozone)

# bagged loess
ll <- matrix(NA, nrow = 10, ncol = 155) #create a matrix
for (i in 1:10) { #resample the dataset 10 times with replacement
    ss <- sample(1:dim(ozone)[1], replace = T)
    ozone0 <- ozone[ss, ] #create a new matrix with the samples
    ozone0 <- ozone0[order(ozone0$ozone), ] #reorder dataset by ozone each time
    loess0 <- loess(temperature ~ ozone, data = ozone0, span = 0.2) #fit a loess curve for each matrix, similar to spline model fits, span is how smooth the measure will be
    ll[i, ] <- predict(loess0, newdata = data.frame(ozone = 1:155)) #predict the outcome for each loess curve
}

plot(ozone$ozone, ozone$temperature, pch = 19, cex = 0.5)
for (i in 1:10) {lines(1:155, ll[i, ], col = "gray", lwd = 2)}
lines(1:155, apply(ll, 2, mean), col = "red", lwd = 2)
# the red line is the bagged loess curve
# bagging estimate has lower variability but similar bias to individual models
# train: bagEarth, treebag, bagFDA

# build your own bagging function in caret
# take the predictors and put them in one data frame
library(party)
predictors = data.frame(ozone = ozone$ozone)
temperature = ozone$temperature
treebag <- bag(predictors, temperature, B = 10, #number of replications
               bagControl = bagControl(fit = ctreeBag$fit, #fit model
                                       predict = ctreeBag$pred, #predict
                                       aggregate = ctreeBag$aggregate)) #method of putting predictions together

plot(ozone$ozone, temperature, col = "lightgrey", pch = 19)
points(ozone$ozone, predict(treebag$fits[[1]]$fit, predictors), pch = 19, col = "red")
points(ozone$ozone, predict(treebag, predictors), pch = 19, col = "blue")

# parts of bagging
ctreeBag$fit
ctreeBag$pred
ctreeBag$aggregate

# bagging is most useful for nonlinear models
# often used with trees - an extension is random forests
# several models use bagging in caret's train function

################################################################################
# random forests
# bootstrap samples, rebuild classification trees for each sample
# at each split, we also bootstrap variable (only a subset of variables are used at each split), which makes a diverse set of trees
# grow multiple trees and then either vote or average
# pros: accuracy
# cons: can be slow, hard to interpret, can overfit (important to cross validate)
data(iris); library(ggplot2)
inTrain <- createDataPartition(y = iris$Species, p = 0.7, list = FALSE)
training <- iris[inTrain, ]
testing <- iris[-inTrain, ]

library(caret)
modFit <- train(Species ~ ., data = training, method = "rf", prox = TRUE)
modFit
# look at a specific tree in the final model fit
library(randomForest)
getTree(modFit$finalModel, k = 2) #the second tree, rows are each split

# class "centers" (centers for predicted values)
library(ggplot2)
irisP <- classCenter(training[, c(3, 4)], training$Species, modFit$finalModel$prox)
irisP <- as.data.frame(irisP)
irisP$Species <- rownames(irisP)
p <- qplot(Petal.Width, Petal.Length, col = Species, data = training)
p + geom_point(aes(x = Petal.Width, y = Petal.Length, col = Species), size = 5, shape = 4, data = irisP)

# predict new values
pred <- predict(modFit, testing)
testing$predRight <- pred == testing$Species
table(pred, testing$Species)

# look at the values that were misclassified
qplot(Petal.Width, Petal.Length, color = predRight, 
      data = testing, main = "newdata Predictions")

# can use the rfcv function to make sure the model is not overfitted

################################################################################
# boosting
# take lots of (possibly) weak predictors, weight them and add them up
# gbm: bossting with trees
# mboost: model based boosting
# ada: statistical boosting based on additive logistic regression
# gamBoost: boosting generalized additive models
library(ISLR); data(Wage); library(ggplot2); library(caret)
library(gbm)
Wage <- subset(Wage, select = -c(logwage)) # remove the outcome variable
inTrain <- createDataPartition(y = Wage$wage, p = 0.7, list = FALSE)
training <- Wage[inTrain, ]
testing <- Wage[-inTrain, ]
modFit <- train(wage ~ ., method = "gbm", data = training, verbose = FALSE) #verbose = false otherwise there's a lot of output
print(modFit)
qplot(predict(modFit, testing), wage, data = testing)

################################################################################
# model based approach
# assume the data follow a probablistic model
# use Bayes' theorem to identify optimal classifiers
# linear discriminant analysis assumes multivariate Gaussian with same covariances
# quadratic discriminant analysis assumes multivariate Gaussian with different covariances
# model based prediction assumes more complicated versions for the covariance matrix
# naive bayes assumes independence between features for model building
data(iris); library(ggplot2)
table(iris$Species)
inTrain <- createDataPartition(y = iris$Species, p = 0.7, list = FALSE)
training <- iris[inTrain, ]
testing <- iris[-inTrain, ]
dim(training); dim(testing)

modlda = train(Species ~ ., data = training, method = "lda") #linear discriminant
modnb = train(Species ~ ., data = training, method = "nb") #naive bayes
plda = predict(modlda, testing)
pnb = predict(modnb, testing)
table(plda, pnb) #agree for all but one value
equalPredictions = (plda == pnb)
qplot(Petal.Width, Sepal.Width, color = equalPredictions, data = testing)

################################################################################
# Quiz 3
library(AppliedPredictiveModeling)
library(caret)
library(ElemStatLearn)
library(pgmm)
library(rpart)

# Q1
library(AppliedPredictiveModeling)
data(segmentationOriginal)
library(caret)
training <- segmentationOriginal[segmentationOriginal$Case == "Train", ]
testing <- segmentationOriginal[segmentationOriginal$Case == "Test", ]
dim(training); dim(testing)
set.seed(125)
modFit <- train(Class ~ ., data = training, method = "rpart")
library(rattle)
fancyRpartPlot(modFit$finalModel)

# Q3
library(pgmm)
data(olive)
olive = olive[, -1]
head(olive)
inTrain <- createDataPartition(y = olive$Area, p = 0.7, list = FALSE)
training <- olive[inTrain, ]
testing <- olive[-inTrain, ]
modFit <- train(Area ~ ., data = training, method = "rpart")
newdata = as.data.frame(t(colMeans(olive)))
predict(modFit, newdata)

# Q4
library(ElemStatLearn)
data(SAheart)
set.seed(8484)
train = sample(1:dim(SAheart)[1],size = dim(SAheart)[1]/2,replace = F)
trainSA = SAheart[train, ]
testSA = SAheart[-train, ]
set.seed(13234)
names(trainSA)
modFit <- train(chd ~ age + alcohol + obesity + tobacco + typea + ldl, data = trainSA, method = "glm", family = "binomial")
pred1 <- predict(modFit, newdata = trainSA)
pred2 <- predict(modFit, newdata = testSA)

missClass = function(values,prediction)
{
    sum(((prediction > 0.5) * 1) != values) / length(values)
    } 
#if the prediction is more than 0.5, give it a value of 1, otherwise give it 0
# compare predictions with the actual test responses
# add up how many of them are different, then divide it by the total number of cases
missClass(trainSA$chd, pred1)
missClass(testSA$chd, pred2)

# Q5
library(ElemStatLearn)
library(randomForest)
data(vowel.train)
data(vowel.test)
vowel.train$y <- as.factor(vowel.train$y)
vowel.test$y <- as.factor(vowel.test$y)
set.seed(33833)
modFit <- randomForest(y ~ ., data = vowel.train)
modFit
varImp(modFit) #variable importance in random forest (Gini)
