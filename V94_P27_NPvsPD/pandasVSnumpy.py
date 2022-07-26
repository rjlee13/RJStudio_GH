#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 10:11:43 2022

@author: rj
"""












"""
NumPy vs Pandas
 (▰˘◡˘▰)


Here is a nice table showing the difference between NumPy and Pandas:
    https://www.geeksforgeeks.org/difference-between-pandas-vs-numpy/ 👉


When using Scikit-learn Machine Learning library to train a model,
you can usually give data in "NumPy" arrays or in "Pandas" DataFrame.



For my own curiosity 😁, I wanted to compare ELAPSED TIME 
to train a model using NumPy vs Pandas 🔥 


In this short video, I use 2 relatively simple datasets:
    IRIS dataset & Boston Housing Price dataset
to train a Linear Regression using NumPy and then Pandas
then finally compare ELAPSED TIME to train model & calculate R-squared value ⏱️



Please 🌟PAUSE🌟 the video any time you want to understand code / output.
"""














# =============================================================================
# Load IRIS data in NumPy and Pandas
# =============================================================================

# load IRIS
from sklearn import datasets
iris_raw = datasets.load_iris()

# save data in NumPy array
import numpy as np
x_iris_np = iris_raw.data
y_iris_np = iris_raw.target

# dimension
x_iris_np.shape # (150, 4)
y_iris_np.shape # (150,)

type(x_iris_np) # numpy.ndarray
type(y_iris_np) # numpy.ndarray



# this time,
# save data in Pandas DataFrame
import pandas as pd
x_iris_pd = pd.DataFrame(data    = iris_raw.data,
                         columns =['sepal_len', 'sepal_wid', 
                                   'petal_len', 'petal_wid'])
y_iris_pd = pd.Series(data = iris_raw.target,
                      name = 'specie')

# dimension
x_iris_pd.shape # (150, 4)
y_iris_pd.shape # (150,)    (of course dimension doesn't change)

type(x_iris_pd) # pandas.core.frame.DataFrame
type(y_iris_pd) # pandas.core.series.Series


'''
So dimension is only 150 samples & 4 features

It's a small dataset! 🌱
'''













# =============================================================================
# Linear Regression with IRIS 
# =============================================================================

# Import what I need
import time  # need for measuring Elaspsed Time
from sklearn.linear_model import LinearRegression # for Linear Regression
lr = LinearRegression()



# Elapsed Time with NUMPY data 
iris_np_time = []
for i in range(50):                      # 50 iterations
    start_np = time.time()               # start timer
    lr.fit(X = x_iris_np,                # Training with NUMPY data
           y = y_iris_np)
    lr.score(x_iris_np, y_iris_np)       # R-sq calculation
    end_np = time.time()                 # end timer
    elapsed_time_np = end_np - start_np  # ELAPSED TIME
    iris_np_time.append(elapsed_time_np)

# Average Elapsed Time with NUMPY data 
sum(iris_np_time) / len(iris_np_time)  # ~ 0.00019 sec



# Elapsed Time with PANDAS data 
iris_pd_time = []
for i in range(50):                      # 50 iterations
    start_pd = time.time()               # start timer
    lr.fit(X = x_iris_pd,                # Training with PANDAS data
           y = y_iris_pd)
    lr.score(x_iris_pd, y_iris_pd)       # R-sq calculation
    end_pd = time.time()                 # end timer
    elapsed_time_pd = end_pd - start_pd  # ELAPSED TIME
    iris_pd_time.append(elapsed_time_pd)

# Average Elapsed Time with PANDAS data 
sum(iris_pd_time) / len(iris_pd_time)  # ~ 0.0007 sec


'''
So with small IRIS data, 
NumPy tends to be more than 3 times faster than Pandas
'''













# =============================================================================
# Load Boston House Price Data 🏘️ in NumPy and Pandas
# =============================================================================

# Load Boston data
boston = datasets.load_boston()



# Prepare Boston NUMPY Data
x_boston_np = boston.data
y_boston_np = boston.target

# dimension
x_boston_np.shape # (506, 13)
y_boston_np.shape # (506,)

type(x_boston_np) # numpy.ndarray
type(y_boston_np) # numpy.ndarray



# Prepare Boston PANDAS Data
x_boston_pd = pd.DataFrame(data    = boston.data,
                           columns = boston.feature_names)
y_boston_pd = pd.Series(data = boston.target,
                        name = 'specie')

# dimension
x_boston_pd.shape # (506, 13)
y_boston_pd.shape # (506,)

type(x_boston_pd) # pandas.core.frame.DataFrame
type(y_boston_pd) # pandas.core.series.Series


'''
So Boston House Price dataset has 
    around 3 times more samples (506 > 150 * 3)
    around 3 times more features (13 >  4 * 3)
than IRIS dataset, which has 150 samples and 4 features
'''












# =============================================================================
# Linear Regression with Boston
# =============================================================================

# Elapsed Time with NUMPY data 
boston_np_time = []
for i in range(50):                        # 50 iterations
    start_np = time.time()                 # start timer
    lr.fit(X = x_boston_np,                # Training with NUMPY data
           y = y_boston_np)
    lr.score(x_boston_np, y_boston_np)     # R-sq calculation
    end_np = time.time()                   # end timer
    elapsed_time_np = end_np - start_np    # ELAPSED TIME
    boston_np_time.append(elapsed_time_np)

# Average Elapsed Time with Numpy data 
sum(boston_np_time) / len(boston_np_time)  # ~ 0.00028 sec



# Elapsed Time with PANDAS data 
boston_pd_time = []
for i in range(50):                        # 50 iterations
    start_pd = time.time()                 # start timer
    lr.fit(X = x_boston_pd,                # Training with PANDAS data
           y = y_boston_pd)
    lr.score(x_boston_pd, y_boston_pd)     # R-sq calculation
    end_pd = time.time()                   # end timer
    elapsed_time_pd = end_pd - start_pd    # ELAPSED TIME
    boston_pd_time.append(elapsed_time_pd)

# Average Elapsed Time with Pandas data 
sum(boston_pd_time) / len(boston_pd_time)  # ~ 0.00082 sec




'''
Again, NumPy is about 3 times faster than Pandas :) 
'''



















"""
This is the end of "NumPy vs Pandas" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""














