#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 15:44:00 2023

@author: rj
"""












"""
KNN Regression
 (▰˘◡˘▰)


K-Neighbors Regressor is a Non-Parametric method for predicting
output values by averaging "K" neighbors' values. 



This video compares 2 different types of "weights" used for 
evaluating K neighbors: 
    1) Uniform
    2) Distance

Uniform  - All points in each neighborhood are weighted equally
Distance - CLOSER neighbors have greater influence than FAR neighbors 



We can clearly notice the difference in outcome prediction depending 
on the choice of the WEIGHTS in the Plots panel 👉
(Visualization code shown in the video)



Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""

















# =============================================================================
# Generate Data
# =============================================================================

# import numpy
import numpy as np


# X, Generate 500 random values between 0 and 10
X = np.sort(10 * np.random.rand(500, 1), axis = 0) # sort from smallest
X[:10]   # first 10 values, almost 0 
X[-10:]  # last  10 values, almost 10


# y, Generate output: trigonometric sine of X + some randomness
y = np.sin(X).ravel()            # sine of X
y += (0.5 - np.random.rand(500)) # give some randomness


# Visualize Generated Data X & y
import matplotlib.pyplot as plt
plt.title("Generated Data")
plt.scatter(X, y,
            label = "data")
plt.legend() # notice the famous trigonometric Sine wave



# =============================================================================
# Train & Test data
# =============================================================================

# import train_test_split
from sklearn.model_selection import train_test_split

# 80% train data / 20% test data
(train_x, test_x, train_y, test_y) = train_test_split(
    X, y,
    train_size   = 0.8)



# train data
train_x.shape # (400, 1)  
train_y.shape # (400,)    ; 80% of 500 is 400

# test data
test_x.shape  # (100, 1)  
test_y.shape  # (100,)    ; 20% of 500 is 100
















# =============================================================================
# KNeighborsRegressor: uniform VS distance
# =============================================================================

# import KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor


# KNeighborsRegressor with 'uniform' weight
knn_uni = KNeighborsRegressor(n_neighbors = 15,
                              weights     = 'uniform')
# fit using train data
knn_uni.fit(X = train_x,
            y = train_y)


# KNeighborsRegressor with 'distance' weight
knn_dis = KNeighborsRegressor(n_neighbors = 15,
                              weights     = 'distance')
# fit using train data
knn_dis.fit(X = train_x,
            y = train_y)



# =============================================================================
# Evaluate 2 KNN regressors
# =============================================================================

# Get prediction using test data (test_x)
knn_uni_pred = knn_uni.predict(X = test_x) # uniform weight
knn_dis_pred = knn_dis.predict(X = test_x) # distance weight


preds   = [knn_uni_pred, knn_dis_pred] # KNN prediction of test data
weights = ['uniform', 'distance']      # 2 KNN weights used 
columns = ['mse', 'mae']               # 2 evaluation metrics 

# create Pandas DataFrame to save 2 KNN evaluation metrics 
import pandas as pd
results = pd.DataFrame(index   = weights,
                       columns =  columns) 
results # filled with NaNs for now


# import MSE MAE
from sklearn.metrics import mean_squared_error, mean_absolute_error

# populate results DataFrame 
for pred, weight in zip(preds, weights):

    mse = mean_squared_error(test_y, pred)  # calculate mse
    mae = mean_absolute_error(test_y, pred) # calculate mae
    
    results.loc[weight]['mse'] = round(mse, 3) # populate mse
    results.loc[weight]['mae'] = round(mae, 3) # populate mae

# check results    
results
'''
both MSE & MAE are smaller when uniform distance is used!
'''


















# =============================================================================
# Visualizing KNN predictions
# =============================================================================

# 2 weights to be compared
weights = ['uniform', 'distance'] 

# for loop for training/predicting 2 KNNs and then visualizing predictions
for i, weight in enumerate(weights):
    
    # KNN regressor with 1 of 2 weights
    knn = KNeighborsRegressor(n_neighbors = 15, 
                              weights     = weight # 1 of 2 weights
                              )
    
    # prediction
    linspace = np.linspace(0, 10, 300)[:, np.newaxis]
    y_hat = knn.fit(X = X, y = y).predict(linspace)
    
    # visualize using prediction
    plt.subplot(2, 1, i+1)
    plt.title(f"KNeighborsRegressor | weight = {weight}")
    plt.scatter(X, y,
                color = "pink",
                label = "data")
    plt.plot(linspace, y_hat,    # prediction, y_hat
             color = "black",
             label = "prediction")
    plt.legend()
    plt.tight_layout()

'''
As you can see, 

distance weight prediction is more "sensitive" as it tries to
come closer to the nearest training neighbors (too much).

Therefore, 'distance' weight tends to overfit more than 'uniform' weight.


I suspect that is the reason why
both MSE & MAE are smaller when uniform distance is used in our example.
'''


















"""
This is the end of "KNN Regressor" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""




















