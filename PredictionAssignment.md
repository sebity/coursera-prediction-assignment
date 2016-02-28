# Prediction Assignment: An Analysis of the Weight Lifting Exercises Dataset
Jan Tatham  
27 February 2016  

## Summary

### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. The participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

### Goal

The goal of this project is to predict the manner in which participants did the exercise. This report describes how the weight lifting data was analysed and the prediction model generated. The prediction model was used successfully to accurately predict all 20 different test cases on the Coursera website.

## Getting and Loading the Data

The training data for this project are available here: [Training Data Set](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

The test data are available here: [Test Data Set](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

### Loading the Data and Required Libraries

Load the required libraries


```r
library(ggplot2)
library(caret)
```

```
## Loading required package: lattice
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(splines)
library(gbm)
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.1
```

```r
library(plyr)
library(MASS)
```


Load the data (i.e. `read.csv()`)


```r
# Check if Data folder exists
if(!dir.exists('./Data')) {
  dir.create('./Data')
}

# Check of pml-training.csv file exists in Data folder
if(file.exists("./Data/pml-training.csv") == FALSE) {
  fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
  download.file(fileUrl, destfile = "./Data/pml-training.csv", mode = "wb")
}

# Check of pml-testing.csv file exists in Data folder
if(file.exists("./Data/pml-testing.csv") == FALSE) {
  fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
  download.file(fileUrl, destfile = "./Data/pml-testing.csv", mode = "wb")
}

# Read in the training and testing datasets
training <- read.csv("./Data/pml-training.csv", header=TRUE, na.strings = c("","NA"))
testing <- read.csv("./Data/pml-testing.csv", header=TRUE, na.strings = c("","NA"))
```



## Data Preparation

### Initial Analysis of the Raw Data


```r
dim(training)
```

```
## [1] 19622   160
```

```r
dim(testing)
```

```
## [1]  20 160
```

There are 160 variables, which is a lot and we need to remove the variables which won't be used in this project.

Looking at the data in the **training** dataset, you can see there are a large number of NA values.  So the next stage is to check for near zero-variance predictors and remove those variables from the model.

### Check for near zero-variance


```r
na_count <- sapply(training, function(n) sum(length(which(is.na(n)))))

nzv <- nearZeroVar(training, saveMetrics = TRUE)

table(nzv$nzv, na_count)
```

```
##        na_count
##          0 19216
##   FALSE 59    58
##   TRUE   1    42
```

From the observations we can see that each column had either no missing values or 19,216 missing values.  Therefore we will create a new dataset called **training_clean** with the variables with a near zero-variance and/or have 19,216 missing values.

### Remove the variables with near zero-variance and high missing value count


```r
# remove variables with a high number of NAs and/or near zero-variances
training_clean <- training[, (!nzv$nzv & na_count < 19216)]

# remove variables which aren't measurements, i.e line number, user name, timestamps, etc.
training_clean <- training_clean[,-c(1:6)]
```

We are now left with 53 variables instead of the 160 variables that we started with.


### Split data to training and testing for cross validation.

Next we will prepare the data for cross validation by splitting the cleaned training dataset into **myTraining** and **myTesting**.  In terms of splitting, we will go with a traditional 70/30 split. 70% for the training data and 30% for the testing data.


```r
set.seed(1234)
# Split the cleaned training data.  Assign 70% for training and 30% for testing.
inTrain <- createDataPartition(training_clean$classe, p=0.7, list=FALSE)
myTraining <- training_clean[inTrain,]
myTesting <- training_clean[-inTrain,]
```


## Model Training

### Fit The Model

Now we will train three different types of models which are, random forest, linear discriminant analysis and boosting with trees.


```r
fitControl <- trainControl(method="cv", number = 5, allowParallel = TRUE)

fitRF <- train(classe ~ ., data=myTraining, method="rf", trControl = fitControl)

fitLDA <- train(classe ~ ., data=myTraining, method="lda", trControl = fitControl)

fitGBM <- train(classe ~ ., data=myTraining, method="gbm", trControl = fitControl, verbose = FALSE)
```


## Model Prediction

### Training Data Prediction

We will now test the accuracy of the training data with the training models. 


```r
predictRF1 <- predict(fitRF, myTraining)
confusionMatrix(predictRF1, myTraining$classe)$overall[1]
```

```
## Accuracy 
##        1
```

```r
predictGBM1 <- predict(fitGBM, myTraining)
confusionMatrix(predictGBM1, myTraining$classe)$overall[1]
```

```
##  Accuracy 
## 0.9747398
```

```r
predictLDA1 <- predict(fitLDA, myTraining)
confusionMatrix(predictLDA1, myTraining$classe)$overall[1]
```

```
##  Accuracy 
## 0.7056854
```

The Random Forest had the highest accuracy with **100%**, followed by the Boosting with Trees at **97.5%** and last with Linear Discriminant Analysis at **70.6%**.

### Test Data Prediction

Next we will test the accuracy of the test data that was a subset of the cleaned training data.


```r
predictRF2 <- predict(fitRF, myTesting)
confusionMatrix(predictRF2, myTesting$classe)$overall[1]
```

```
##  Accuracy 
## 0.9940527
```

```r
predictGBM2 <- predict(fitGBM, myTesting)
confusionMatrix(predictGBM2, myTesting$classe)$overall[1]
```

```
##  Accuracy 
## 0.9636364
```

```r
predictLDA2 <- predict(fitLDA, myTesting)
confusionMatrix(predictLDA2, myTesting$classe)$overall[1]
```

```
##  Accuracy 
## 0.7048428
```

We got similar results to the training data.  Random Forest: **99.4%**, Boosting: **96.4%** and Linear Discriminant Analysis: **70.5%**.


## Variable Importance

We will now list the ten most important variables in the model.


```r
var_imp <- varImp(fitRF)$importance

var_imp[head(order(unlist(var_imp), decreasing = TRUE), 10L), , drop = FALSE]
```

```
##                     Overall
## roll_belt         100.00000
## pitch_forearm      60.91394
## yaw_belt           57.96347
## magnet_dumbbell_y  44.86773
## pitch_belt         43.85362
## magnet_dumbbell_z  43.53392
## roll_forearm       40.05661
## accel_dumbbell_y   20.55266
## roll_dumbbell      18.17853
## magnet_dumbbell_x  17.23695
```

From the model we can see that **Roll Belt** is the most important by a large margin, followed by Pitch Forearm and then Yaw Belt. 

## Final Tests

We will now test out models on the original testing dataset using our random forest model.


```r
predictRF3 <- predict(fitRF, testing)
predictRF3
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

These results were submitted to the Course Prediction Quiz and received 20/20, a 100% success rate.



### Citations

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
