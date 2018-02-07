# [Taxi trip time prediction](https://www.kaggle.com/c/nyc-taxi-trip-duration)

This is a Kaggle competition.

## Introduction

For an intelligent transportation system, when we hope to go somewhere, it is possible to have different routes. But which one is fastest
would be very important to us. This competition requires us to predict the trip time given the starting point and ending point.

## Data observation
![data information](https://github.com/SeanCsc/Taxi-trip-time-prediction/blob/master/other/data_info.jpg)


## Data Preprocessing and Exploration

For the data preprocessing and exploration

  1. Load the csv files
  2. Fill in the missing values. Most of the missing value of features are because of lack of that feature, such as fence, I filled them with None. For some numerical feature, I fill them with the mean value of the neighbors or 0.
  3. For non-numerical features, use get_dummies to transform them to 0/1 value.
  
## Feature Engineering
  1. Feature importance. Use Lasso to measure the importance of feature and drop the feature with 0 variance.
  2. Add related features like total square
  3. Standarization and Normalization
  
## Model Selection

Because this is a regression problem. So I decided to use linear model, like ridge and lasso regression, elastic-net. Besides, I use Tree model like XGBT. 

## Model Evaluation
I use 5-fold to do cross-validation and negative MSE as the measurement of the performance for model.

## Model Training
For the linear model, I use cross-validation to choose the best parameter among 5 candidates. For tree models, I use grid search twice to determine the max depth, number of trees. 


## Ensemble Generation

Ensemble Learning refers to the technique of combining different models. It reduces both bias and variance of the final model, thus increasing the score and reducing the risk of overfitting. I mainly use the idea of stack, which is use the output of basic models as the input for the final model. In terms of basic models, I choose to average the models elastic net, GBoost and kernel-ridge. For the meta model, I use the linear model lasso.
[stack in practice](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/)

## Results

My submission is currently ranked in the Top 10% on Kaggle and this can certainly be improved.

## Tools Utilized

####Stack:

* python

####Modeling:

* numpy
* scipy
* pandas
* scikit-learn
* xgboost
* model-selection

####Statistics observation:

* matplotlib
* seaborn

## Update

v1.2 Skewness analysis
For linear regression model, it assumes that the residual satisfies the normal distribution. Therefore, skewness should be avoided. Besides, for price data, it is more sensable to use log. 
[deal with skewness](https://becominghuman.ai/how-to-deal-with-skewed-dataset-in-machine-learning-afd2928011cc)

v1.3 For XGBT model parameter tuning, set the default parameters and use ROC-AUC to firstly have a sense about overfitting or underfitting. Then change related parameters.
