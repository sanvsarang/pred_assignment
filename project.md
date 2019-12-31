---
title: "Prediction Assignment"
author: "Sanil Sarang"
date: "December 28, 2019"
output: 
  html_document:
    keep_md: true
---
##Summary

The objective is to determne if give exercise is done properly or not based on wearables accelerometer data. Based on accerlerometer readings, the training data is classified into 5 classes - A,B,C,D, and E. A is the the proper way of doing exercise while the rest four are the improper ways of doing the exercise. In this report a model will be developed to correctly classify these 5 classes.  



####Loading R libraries

```r
library(caret)
library(data.table)
library(dplyr)
```

## Data

Data is collected from six participants' wearables data - acceloremeters on the belt, forearm, arm, and dumbell for every time window. The participants do different weight-lefting exercises classified as A,B,C,D, and E. A is the the proper way of doing exercise while the rest four are the improper ways of doing the exercise. For each time window there is a summary data consisting of avg, stddev, etc. Data is split into training and testing data.

####Reading data

```r
training_dt <- fread("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", stringsAsFactors = T)
testing_dt <- fread("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", stringsAsFactors = T)
training <- data.frame(training_dt)
testing <- data.frame(testing_dt)
```

####Removing derived summary rows, summary attributes (columns), timestamps row, time frame windows as they should not impact classification problem at hand


```r
training <- training %>%
  filter(new_window == "no") %>%
  select(-starts_with("kurtosis_")) %>%
  select(-starts_with("skewness_")) %>%
  select(-starts_with("max_")) %>%
  select(-starts_with("min_")) %>%
  select(-starts_with("amplitude_")) %>%
  select(-starts_with("var_")) %>%
  select(-starts_with("avg_")) %>%
  select(-starts_with("stddev_")) %>%
  select(-starts_with("V1")) %>%
  select(-starts_with("raw_")) %>%
  select(-starts_with("new_")) 
  
testing <- testing %>%
  filter(new_window == "no") %>%
  select(-starts_with("kurtosis_")) %>%
  select(-starts_with("skewness_")) %>%
  select(-starts_with("max_")) %>%
  select(-starts_with("min_")) %>%
  select(-starts_with("amplitude_")) %>%
  select(-starts_with("var_")) %>%
  select(-starts_with("avg_")) %>%
  select(-starts_with("stddev_")) %>%
  select(-starts_with("V1")) %>%
  select(-starts_with("raw_")) %>%
  select(-starts_with("new_")) 
```

## Modelling decisions

### Data splitting
The original data is already split into training and testing data. For modelling purpose, have further split the data into training-train data and training-test data. The model will be tuned on training-train data and then vaildated on training-test data. Final evaluation will be done on the testing data.  

### Model selection 
Need a powerful classification model that requires good accuracy. Choosing random forest as it apples ensemble training. Since random forest provided the required accuracy other models were not tried. 

### Preprocessing
Apart from remoing data as described in remove data section above, no other preprocessing was applied. Decision tree based models typically not require preprocessing.

### Cross validation
Repeated cross validation was used while tuning the random forest model to determine the best accuracy. This is K-fold (10 fold) cross validation 3 times. For tuning data is resampled multiple times to determne which mtry parameter value for random forest is optimal to achieve the highest accuracy. mtry is the number of attributes(columns) used to construct each tree in random forest. For each repeat, 10 fold cv is applied - the training-train data is split into 10 samples and 1 sample is hold out for validating accuracy. 

### Out of Sample error


```r
set.seed(323)
intrain <- createDataPartition(training$classe,p = 0.7, list = F)
training_train <- training[intrain,]
training_test <- training[-intrain,]


tuning_grid <- expand.grid(.mtry =c(15,30,45))
tr_control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

set.seed(323)
rf_mod <- train(classe ~ .,data = training_train, method = "rf", tuneGrid = tuning_grid, trControl = tr_control )

rf_mod
```

```
## Random Forest 
## 
## 13453 samples
##    55 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold, repeated 3 times) 
## Summary of sample sizes: 12107, 12108, 12108, 12107, 12109, 12107, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##   15    0.9977452  0.9971475
##   30    0.9981415  0.9976489
##   45    0.9976707  0.9970531
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 30.
```


```r
rf_pred <- predict(rf_mod, training_test)

cfm <- confusionMatrix(rf_pred, training_test$classe)

cfm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1641    1    0    0    0
##          B    0 1112    0    0    0
##          C    0    2 1005    2    0
##          D    0    0    0  942    4
##          E    0    0    0    0 1054
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9984         
##                  95% CI : (0.997, 0.9993)
##     No Information Rate : 0.2847         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.998          
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9973   1.0000   0.9979   0.9962
## Specificity            0.9998   1.0000   0.9992   0.9992   1.0000
## Pos Pred Value         0.9994   1.0000   0.9960   0.9958   1.0000
## Neg Pred Value         1.0000   0.9994   1.0000   0.9996   0.9992
## Prevalence             0.2847   0.1935   0.1744   0.1638   0.1836
## Detection Rate         0.2847   0.1930   0.1744   0.1635   0.1829
## Detection Prevalence   0.2849   0.1930   0.1751   0.1642   0.1829
## Balanced Accuracy      0.9999   0.9987   0.9996   0.9985   0.9981
```


```r
test_rf_pred <- predict(rf_mod, testing)

test_rf_pred
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
