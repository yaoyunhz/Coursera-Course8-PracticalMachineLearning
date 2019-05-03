# Course8 Practical Machine Learning

# Week4
library(ElemStatLearn); data(prostate)
str(prostate)

small = prostate[1:5, ]
lm(lpsa ~ ., data = small)

################################################################################
# comining predictors
library(ISLR); data(Wage); library(ggplot2); library(caret)
Wage <- subset(Wage, select = -c(logwage))

# create a building data set and validation set
inBuild <- createDataPartition(y = Wage$wage, p = 0.7, list = FALSE)
validation <- Wage[-inBuild, ]
buildData <- Wage[inBuild, ]

inTrain <- createDataPartition(y = buildData$wage, p = 0.7, list = FALSE)
training <- buildData[inTrain, ]
testing <- buildData[-inTrain, ]

dim(training)
dim(testing)
dim(validation)

# build two different models
mod1 <- train(wage ~ ., method = "glm", data = training)
mod2 <- train(wage ~ ., method = "rf", data = training, 
              trControl = trainControl(method = "cv"), number = 3)

# plot the predictions on the testing set
pred1 <- predict(mod1, testing)
pred2 <- predict(mod2, testing)
qplot(pred1, pred2, color = wage, data = testing)

# fit a model that combines predictors
predDF <- data.frame(pred1, pred2, wage = testing$wage)
head(predDF)
combModFit <- train(wage ~ ., method = "gam", data = predDF)
combPred <- predict(combModFit, predDF)

# testing errors
sqrt(sum((pred1 - testing$wage) ^ 2))
sqrt(sum((pred2 - testing$wage) ^ 2))
sqrt(sum((combPred - testing$wage) ^ 2))

# predict on validation dataset
pred1V <- predict(mod1, validation)
pred2V <- predict(mod2, validation)
predVDF <- data.frame(pred1 = pred1V, pred2 = pred2V)
combPredV <- predict(combModFit, predVDF) #predictions from two models
sqrt(sum((pred1V - validation$wage) ^ 2))
sqrt(sum((pred2V - validation$wage) ^ 2))
sqrt(sum((combPredV - validation$wage) ^ 2))

################################################################################
# forecasting
library(quantmod)
from.dat <- as.Date("01/01/08", format = "%m/%d/%y")
to.dat <- as.Date("12/31/13", format = "%m/%d/%y")
getSymbols("GOOG", src = "google", from = from.dat, to = to.dat) #load stock

# summarize monthly and store as time series
mGoog <- to.monthly(GOOG) #convert to a monthly time series
googOpen <- Op(mGoog) #take just opening information
ts1 <- ts(googOpen, frequency = 12) #time series
plot(ts1, xlab = "Years + 1", ylab = "GOOG")

# decompose a time series into parts
plot(decompose(ts1), xlab = "Years + 1")

# training and test sets
ts1Train <- window(ts1, start = 1, end = 5) #consecutive time points
ts1Test <- window(ts1, start = 5, end = (7 - 0.01))
ts1Train

# forecasting: simple moving average
plot(ts1Train)
lines(ma(ts1Train, order = 3), col = "red")

# forecasting: exponential smoothing
ets1 <- ets(ts1Train, model = "MMM")
fcast <- forecast(ets1)
plot(fcast)
lines(ts1Test, col = "red")
# get the accuracy
accuracy(fcast, ts1Test) #RMSE

################################################################################
# unsupervised prediction
data(iris); library(ggplot2)
inTrain <- createDataPartition(y = iris$Species, p = 0.7, list = FALSE)
training <- iris[inTrain, ]
testing <- iris[-inTrain, ]
dim(training); dim(testing)

# cluster with k-means
kMeans1 <- kmeans(subset(training, select = -c(Species)), centers = 3) #create 3 clusters ignoring the species information
training$clusters <- as.factor(kMeans1$cluster)
qplot(Petal.Width, Petal.Length, color = clusters, data = training)
table(kMeans1$cluster, training$Species)

# build predictor
modFit <- train(clusters ~ ., 
                data = subset(training, select = -c(Species)), method = "rpart")
table(predict(modFit, training), training$Species)

# apply on test
testClusterPred <- predict(modFit, testing)
table(testClusterPred, testing$Species)

################################################################################
# quiz4
library(AppliedPredictiveModeling)
library(caret)
library(ElemStatLearn)
library(pgmm)
library(rpart)
library(gbm)
library(lubridate)
library(forecast)
library(e1071)

# Q1
library(ElemStatLearn)
data(vowel.train)
data(vowel.test)
vowel.train$y <- as.factor(vowel.train$y)
vowel.test$y <- as.factor(vowel.test$y)
set.seed(33833)
#random forest
mod1 <- train(y ~ ., method = "rf", data = vowel.train)
pred1 <- predict(mod1, newdata = vowel.test)
confusionMatrix(pred1, vowel.test$y)$overall[["Accuracy"]]
#boosting
mod2 <- train(y ~ ., method = "gbm", data = vowel.train, verbose = FALSE)
pred2 <- predict(mod2, newdata = vowel.test)
confusionMatrix(pred2, vowel.test$y)$overall[["Accuracy"]]
# fit a model that combines predictors
predDF <- data.frame(pred1, pred2, y = vowel.test$y)
combMod <- train(y ~ ., method = "gam", data = predDF)
combPred <- predict(combMod, predDF)
confusionMatrix(combPred, predDF$y)$overall[["Accuracy"]]
table(pred1, pred2)

# Q2
library(caret)
library(gbm)
set.seed(3433)
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain, ]
testing = adData[-inTrain, ]
set.seed(62433)
# random forest
modRF <- train(diagnosis ~ ., method = "rf", data = training)
predRF <- predict(modRF, newdata = testing)
confusionMatrix(predRF, testing$diagnosis)$overall[["Accuracy"]]
# boosted trees
modGBM <- train(diagnosis ~ ., method = "gbm", data = training, verbose = FALSE)
predGBM <- predict(modGBM, newdata = testing)
confusionMatrix(predGBM, testing$diagnosis)$overall[["Accuracy"]]
# linear discriminant
modLDA <- train(diagnosis ~ ., method = "lda", data = training)
predLDA <- predict(modLDA, newdata = testing)
confusionMatrix(predLDA, testing$diagnosis)$overall[["Accuracy"]]
# stack the predictions together using random forests
predDF <- data.frame(predRF, predGBM, predLDA, diagnosis = testing$diagnosis)
combMod <- train(diagnosis ~ ., method = "rf", data = predDF)
combPred <- predict(combMod, predDF)
confusionMatrix(combPred, predDF$diagnosis)$overall[["Accuracy"]]

# Q3
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

library(elasticnet)
set.seed(233)
modFit <- train(CompressiveStrength ~ ., method = "lasso", data = training)
plot(modFit$finalModel, use.color = T) #wrong
plot(modFit$finalModel, xvar = "penalty", use.color = T)

# Q4
library(lubridate) # For year() function below
library(forecast)
setwd("C:/Users/yun.yao/Desktop/data science/Course8 Practical Machine Learning")
dat = read.csv("gaData.csv")
dat$date <- as.Date(dat$date)
training = dat[year(dat$date) < 2012, ]
testing = dat[(year(dat$date)) > 2011, ]
tstrain = ts(training$visitsTumblr)
mod <- bats(tstrain)
fcast <- forecast(mod)

# Q5
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain, ]
testing = concrete[-inTrain, ]
set.seed(325)
mod <- svm(CompressiveStrength ~ ., data = training)
pred <- predict(mod, newdata = testing)
sqrt(mean((testing$CompressiveStrength - pred) ^ 2)) #RMSE 6.72
