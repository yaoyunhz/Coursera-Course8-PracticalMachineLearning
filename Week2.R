
# Course8 Practical Machine Learning

# Week2

# SPAM example: data splitting
library(caret); library(kernlab); data(spam)
head(spam)
#split based on type
inTrain <- createDataPartition(y = spam$type, p = 0.75, list = FALSE) 
training <- spam[inTrain, ]
testing <- spam[-inTrain, ]
dim(training)

# fit a model
set.seed(32343)
modelFit <- train(type ~ ., data = training, method = "glm")
modelFit
modelFit$finalModel

# prediction
predictions <- predict(modelFit, newdata = testing)
predictions
confusionMatrix(predictions, testing$type)

# cross validation - k-fold
set.seed(32323)
# list = TRUE means each fold is a list (you can also have vector or matrix)
folds <- createFolds(y = spam$type, k = 10, list = TRUE, returnTrain = TRUE)
sapply(folds, length)
folds[[1]][1:10]
# return the test set
folds <- createFolds(y = spam$type, k = 10, list = TRUE, returnTrain = FALSE)
sapply(folds, length)
folds[[1]][1:10]

# cross validation - resampling 
# resample with replacements (you may get the same value back)
set.seed(32323)
folds <- createResample(y = spam$type, times = 10, list = TRUE)
sapply(folds, length)
folds[[1]][1:10]

# cross validation - time slices
set.seed(32323)
tme <- 1:1000
# take continuous variables in time, 20 samples in each window for training, and the next 10 in the testing window
folds <- createTimeSlices(y = tme, initialWindow = 20, horizon = 10)
names(folds)
folds$train[[1]]
folds$test[[1]]

# train options
args(trainControl)

################################################################################
# plotting predictions
# example: wage data
library(ISLR); library(ggplot2); library(caret)
data(Wage)
summary(Wage)

# get training/test sets
inTrain <- createDataPartition(y = Wage$wage, p = 0.7, list = FALSE)
training <- Wage[inTrain, ]
testing <- Wage[-inTrain, ]
dim(training)
dim(testing)

# feature plot (caret package)
featurePlot(x = training[, c("age", "education", "jobclass")],
            y = training$wage,
            plot = "pairs")

# Qplot (ggplot2 package)
qplot(age, wage, data = training)
qplot(age, wage, color = jobclass, data = training)

# add regression smoothers (ggplot2 package)
qq <- qplot(age, wage, colour = education, data = training)
qq + geom_smooth(method = "lm", formula = y ~ x)

# cut2, making factors (Hmisc package)
library(Hmisc)
cutWage <- cut2(training$wage, g = 3)
table(cutWage)

# boxplots with cut2
p1 <- qplot(cutWage, age, data = training, fill = cutWage, geom = c("boxplot"))
p1

# boxplots with points overlayed (gridExtra package)
library(gridExtra)
p2 <- qplot(cutWage, age, data = training, fill = cutWage, geom = c("boxplot", "jitter"))
grid.arrange(p1, p2, ncol = 2)

# table
t1 <- table(cutWage, training$jobclass)
t1
prop.table(t1) #proportion in each group
prop.table(t1, 1) #row
prop.table(t1, 2) #column

# density plot
qplot(wage, color = education, data = training, geom = "density")

################################################################################
# preprocessing
library(caret); library(kernlab); data(spam)
inTrain <- createDataPartition(y = spam$type, p = 0.75, list = FALSE) 
training <- spam[inTrain, ]
testing <- spam[-inTrain, ]
hist(training$capitalAve, main = "", xlab = "ave. capital run length")
# very skewed and needs preprocessing
mean(training$capitalAve)
sd(training$capitalAve) #highly variable

# standardizing
trainCapAve <- training$capitalAve
trainCapAveS <- (trainCapAve - mean(trainCapAve)) / sd(trainCapAve)
mean(trainCapAveS)
sd(trainCapAveS) #reduce variability

# standardizing - test set
# apply the mean and sd from the training
testCapAve <- testing$capitalAve
testCapAveS <- (testCapAve - mean(trainCapAve)) / sd(trainCapAve)
mean(testCapAveS)
sd(testCapAveS)

# standardizing - preProcess function
# enter all variables except the actual outcome (type)
preObj <- preProcess(training[, -58], method = c("center", "scale"))
trainCapAveS <- predict(preObj, training[, -58])$capitalAve
mean(trainCapAveS)
sd(trainCapAveS)
# can apply the processed training object to the testing set
testCapAveS <- predict(preObj, testing[, -58])$capitalAve
mean(testCapAveS)
sd(testCapAveS)

# pass the preProcess function directly to the train command as an argument
set.seed(32343)
modelFit <- train(type ~., data = training,
                  preProcess = c("center", "scale"), method = "glm")
modelFit

# other kinds of transformation: Box-Cox transforms
# box-cox takes continuous data and try to make them look like normal data
preObj <- preProcess(training[, -58], method = c("BoxCox"))
trainCapAveS <- predict(preObj, training[, -58])$capitalAve
par(mfrow = c(1, 2)); hist(trainCapAveS); qqnorm(trainCapAveS)
# does not take care of all problems, does not take care of values that are repeated


# imputing data: k-nearest neighbor's imputation
set.seed(13343) #randomized algorithm
# make some values NA
training$capAve <- training$capitalAve
selectNA <- rbinom(dim(training)[1], size = 1, prob = 0.05) == 1 #randomly generate missing values
training$capAve[selectNA] <- NA

# impute and standardize
preObj <- preProcess(training[, -58], method = "knnImpute") #k-nearest neighbor's
capAve <- predict(preObj, training[, -58]$capAve)

# standardize true values
capAveTruth <- training$capitalAve
capAveTruth <- (capAveTruth - mean(capAveTruth)) / sd(capAveTruth)

quantile(capAve - capAveTruth)
quantile((capAve - capAveTruth)[selectNA]) #just the values that imputed
quantile((capAve - capAveTruth)[!selectNA])

# training and test must be processed in the same way (builit in caret)
# test transformations will likely be imperfect
# careful when transforming factor variables
# most machine learning algorithms are built for either binary outcomes (not transformed) or continous outcomes

################################################################################
# covariate creation
library(kernlab); data(spam)
spam$capitalAveSq <- spam$capitalAve ^ 2

library(ISLR); library(caret); data(Wage)
inTrain <- createDataPartition(y = Wage$wage, p = 0.7, list = FALSE)
training <- Wage[inTrain, ]; testing <- Wage[-inTrain, ]

# turn factor variables to dummy variables (quantitative)
table(training$jobclass)
dummies <- dummyVars(wage ~ jobclass, data = training)
head(predict(dummies, newdata = training)) #two new variables

# remove zero covariates
nsv <- nearZeroVar(training, saveMetrics = TRUE) #identity bad predictors
nsv

# spline basis (allow curvy model fitting)
library(splines)
bsBasis <- bs(training$age, df = 3) #bs creates a polynomial variable (3rd degree polynomial)
bsBasis #three new variables: scaled age, age squared, age cubed
lm1 <- lm(wage ~ bsBasis, data = training) #all three predictors from the polynomial model
plot(training$age, training$wage, pch = 19, cex = 0.5) #there appears to be a curvy relationship
points(training$age, predict(lm1, newdata = training), col = "red", pch = 19, cex = 0.5)
predict(bsBasis, age = testing$age) #same procedure for the testing set

################################################################################
# preprocessing with principal components analysis (PCA)
library(caret); library(kernlab); data(spam)
inTrain <- createDataPartition(y = spam$type, p = 0.75, list = FALSE)
training <- spam[inTrain, ]
testing <- spam[-inTrain, ]

M <- abs(cor(training[, -58])) #leave out the outcome column, calculate the correlations between all other predictors and take absolute values
diag(M) <- 0 #reset correlation with itself from 1 to 0
which(M > 0.8, arr.ind = T) #which variables have correlations higher than 0.8

# correlated predictors
names(spam)[c(34, 32)]
plot(spam[, 34], spam[, 32])
# may not be useful to include both variables
# basic PCA ideas: a weighted combination of these two predictors may be better
# pick the combination to capture the most informaiton possible
# benefits: reduce number of predictors, reduce noise (due to averaging)

# rotate the plot
X <- 0.71 * training$num415 + 0.71 * training$num857 #sum
Y <- 0.71 * training$num415 - 0.71 * training$num857 #difference
plot(X, Y) #adding the variables together provides more information than subtracting, so we may want to use the sum as a predictor

# principal components - prcomp
smallSpam <- spam[, c(34, 32)]
prComp <- prcomp(smallSpam)
plot(prComp$x[, 1], prComp$x[, 2]) #the first principal component looks like adding and the second principal component looks like subtracting
prComp$rotation

typeColor <- ((spam$type == "spam") * 1 + 1) #black if not spam, red if spam
prComp <- prcomp(log10(spam[, -58] + 1)) #calculate principal components for the entire dataset, log10 + 1 to make the dataset more normal
plot(prComp$x[, 1], prComp$x[, 2], col = typeColor, xlab = "PC1", ylab = "PC2")
#principal component 1 explains most, principal component 2 explains second most
#reduce the size of dataset while still capture a large amount of variation

# PCA with caret
preProc <- preProcess(log10(spam[, -58] + 1), method = "pca", pcaComp = 2) #tell it the method is pca and the number of principal components is 2
spamPC <- predict(preProc, log10(spam[, -58] + 1))
plot(spamPC[, 1], spamPC[, 2], col = typeColor)

preProc <- preProcess(log10(training[, -58] + 1), method = "pca", pcaComp = 2)
trainPC <- predict(preProc, log10(training[, -58] + 1))
modelFit <- train(training$type ~ ., method = "glm", data = trainPC)
testPC <- predict(preProc, log10(testing[, -58] + 1))
confusionMatrix(testing$type, predict(modelFit, testPC))

# alternatives (sets # of PCs)
modelFit <- train(training$type ~ ., method = "glm", preProcess = "pca", data = training)
confusionMatrix(testing$type, predict(modelFit, testing))

# predicting with regression
library(caret); data("faithful"); set.seed(333)
inTrain <- createDataPartition(y = faithful$waiting, p = 0.5, list = FALSE)
trainFaith <- faithful[inTrain, ]; testFaith <- faithful[-inTrain, ]
head(trainFaith)
plot(trainFaith$waiting, trainFaith$eruptions, 
     pch = 19, col = "blue", xlab = "Waiting", ylab = "Duration")
# fit a linear model
lm1 <- lm(eruptions ~ waiting, data = trainFaith)
summary(lm1)
lines(trainFaith$waiting, lm1$fitted, lwd = 3)
# predict a new value
coef(lm1)[1] + coef(lm1)[2] * 80
newdata <- data.frame(waiting = 80)
predict(lm1, newdata)
# plot predictions - training and test
par(mfrow = c(1, 2))
plot(trainFaith$waiting, trainFaith$eruptions, 
     pch = 19, col = "blue", xlab = "Waiting", ylab = "Duration")
lines(trainFaith$waiting, predict(lm1), lwd = 3)
plot(testFaith$waiting, testFaith$eruptions, 
     pch = 19, col = "blue", xlab = "Waiting", ylab = "Duration")
lines(testFaith$waiting, predict(lm1, newdata = testFaith), lwd = 3)
# calculate RMSE (root mean squared error) on training
sqrt(sum((lm1$fitted - trainFaith$eruptions) ^ 2))
# calculate RMSE on test
sqrt(sum((predict(lm1, newdata = testFaith) - testFaith$eruptions) ^ 2))
# test data set error is always larger than the training data set error
# prediction intervals
pred1 <- predict(lm1, newdata = testFaith, interval = "prediction")
ord <- order(testFaith$waiting) #order the values from test dataset
plot(testFaith$waiting, testFaith$eruptions, pch = 19, col = "blue")
matlines(testFaith$waiting[ord], pred1[ord, ], 
         type = "l", col = c(1, 2, 2), lty = c(1, 1, 1), lwd = 3) #intervals that capture where the predicted values to land (range of possible predictions)

# same process with caret
modFit <- train(eruptions ~ waiting, data = trainFaith, method = "lm")
summary(modFit$finalModel)

################################################################################
# predicting with regression: multiple covariates
library(ISLR); library(ggplot2); library(caret)
data(Wage)
Wage <- subset(Wage, select = -c(logwage)) #exclude the outcome variable
summary(Wage)

inTrain <- createDataPartition(y = Wage$wage, p = 0.7, list = FALSE)
training <- Wage[inTrain, ]; testing <- Wage[-inTrain, ]
dim(training); dim(testing)
# feature plot
featurePlot(x = training[, c("age", "education", "jobclass")], 
            y = training$wage, plot = "pairs")
# plot age vs. wage
qplot(age, wage, data = training)
# plot age vs. wage, color by jobclass
qplot(age, wage, color = jobclass, data = training) #jobclass information can predict some variability
# plot age vs. wage, color by education
qplot(age, wage, color = education, data = training) #advanced degree exaplains some variability
# fit a linear model with multiple variables (indicator variables for factor variables jobclass and education)
modFit <- train(wage ~ age + jobclass + education, method = "lm", data = training)
finMod <- modFit$finalModel
print(modFit)
# diagnostics
plot(finMod, 1, pch = 19, cex = 0.5, col = "#00000010") # residuals
# color by variables not used in the model
qplot(finMod$fitted, finMod$residuals, color = race, data = training)
# plot by index
plot(finMod$residuals, pch = 19) #which row of the data, if there's a trend with row number, then it indicates there's a variable missing (time, age, etc.)
# predicted vs. truth in test set
pred <- predict(modFit, testing)
qplot(wage, pred, color = year, data = testing) #ideally a 45 degree line
# if you want to use all covariates
modFitAll <- train(wage ~ ., data = training, method = "lm")
pred <- predict(modFitAll, testing)
qplot(wage, pred, data = testing)

################################################################################
# Quiz 2
# Q1
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
testIndex = createDataPartition(diagnosis, p = 0.50,list = FALSE)
training = adData[-testIndex,]
testing = adData[testIndex,]

# Q2
library(AppliedPredictiveModeling)
library(Hmisc)
library(ggplot2)
data(concrete)
library(caret)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]
head(training)
# Cement
cutCement <- cut2(training$Cement, g = 2)
g <- ggplot(data = training, aes(x = row.names(training), y = CompressiveStrength, color = cutCement))
g + geom_point() + scale_color_brewer()
# BlastFurnaceSlag
cutBlast <- cut2(training$BlastFurnaceSlag, g = 2)
g <- ggplot(data = training, aes(x = row.names(training), y = CompressiveStrength, color = cutBlast))
g + geom_point()
# FlyAsh
cutFly <- cut2(training$FlyAsh, g = 4)
g <- ggplot(data = training, aes(x = row.names(training), y = CompressiveStrength, color = cutFly))
g + geom_point()
# Water
cutWater <- cut2(training$Water, g = 2)
g <- ggplot(data = training, aes(x = row.names(training), y = CompressiveStrength, color = cutWater))
g + geom_point()
# Age
cutAge <- cut2(training$Age, g = 4)
g <- ggplot(data = training, aes(x = row.names(training), y = CompressiveStrength, color = cutAge))
g + geom_point()

# Q3
library(AppliedPredictiveModeling)
data(concrete)
library(caret)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]
hist(training$Superplasticizer)

# Q4
library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]
names(training)
subset <- training[, c(58:69)]
prComp <- prcomp(subset)
prComp
# set threshold to 90%
preProc <-  preProcess(subset, method = c("center", "scale", "pca"), thresh = 0.9, verbose = T)
preProc

# Q5
library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]
subset = training[, c(1, 58:69)]
# model1 (predictors as they are)
model1 <- train(diagnosis ~ ., data = subset, method = "glm")
prediction1 <- predict(model1, newdata = testing)
confusionMatrix(prediction1, testing$diagnosis)
# model2 (pca)
preProc <- preProcess(subset[, -1], method = "pca", thresh = 0.8)
trainPC <- predict(preProc, subset[, -1])
model2 <- train(y = subset$diagnosis, method = "glm", x = trainPC)
prediction2 <- predict(preProc, testing[, -1])
confusionMatrix(testing$diagnosis, predict(model2, prediction2))
