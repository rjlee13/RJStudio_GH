#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 21:13:37 2022

@author: rj
"""









"""
sklearn.pipeline
 (▰˘◡˘▰)

In this video, I want to show you a very convenient/useful 
regression technique, called Pipeline()

To train a polynomial regression, for example, your data 
first needs to be 'transformed', and then implement fit().

Instead of conducting 'transform' & 'fit' SEPARATELY,
they can be 'PIPED together' using sklearn.pipeline 🚀


Note: I will skim through Polynomial Regression code quickly
because I already explained it in detail in another video!
(Video Link in Description)


Please 🌟PAUSE🌟 the video any time you want to understand code / output.
"""

# =============================================================================
# Modules Used
# =============================================================================

from sklearn.pipeline import Pipeline # <-- FOCUS of video

# whole bunch of sklearn :) 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import pandas as pd

















# =============================================================================
# (Quick) Load & Visualize IRIS data 
# =============================================================================

# load IRIS data
iris_data = datasets.load_iris()

# Create IRIS Pandas DataFrame  
iris_pd = pd.DataFrame(
    data    = iris_data.data,                 # IRIS data
    columns = ['sepal_length', 'sepal_width', # custom column names
               'petal_length', 'petal_width']
    )

# visualize petal_width vs petal_length
plt.scatter('petal_length', # x
            'petal_width',  # y
            data = iris_pd)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

'''
According to the visualization,
Petal Width tends to get longer as Petal Length gets longer.

Just to demonstrate Pipeline technique using Polynomial Regression,
Petal Length is used to predict Petal Width 🌱
'''

















# =============================================================================
# Polynomial Regression degree 2 (quick review) 
# NOT using sklearn.pipeline
# =============================================================================
'''
As I mentioned in the beginning,
I quickly skim Polynomial Regression steps
because I already covered it in detail in another video~ 🫠
(Video Link in Description)
'''

# instantiate Polynomial degree 2
poly_deg2 = PolynomialFeatures(degree = 2)

# generate Polynomial degree 2 transformation of Petal Length
petal_length = iris_pd.petal_length.values
petal_length_poly = poly_deg2.fit_transform(petal_length.reshape(-1,1))

# instantiate LinearRegression class
poly2Reg = LinearRegression()

# fit 
poly2Reg.fit(X = petal_length_poly,    # poly degree 2 transformed Petal Length
             y = iris_pd.petal_width   # predicting Petal Width
             )

# prediction of Petal Width using Polynomial Model
petal_width_pred = poly2Reg.predict(X = petal_length_poly)
petal_width_pred

# Caluculate MSE & R-squared
mse  = mean_squared_error(iris_pd.petal_width, petal_width_pred)
rsq = poly2Reg.score(petal_length_poly, iris_pd.petal_width)

print(f"MSE: {mse.round(5)} and R-squared: {rsq.round(5)}")
      # MSE: 0.04203 and R-squared: 0.92717














# =============================================================================
# sklearn.pipeline
# =============================================================================
'''
This time, let me repeat Polynomial degree 2 Regression fit
using sklearn.pipeline 🔗

As we saw earlier,
1st step is to conduct poly degree 2 transformation of Petal Length 
2nd step is to fit LinearRegression with 'transformed' Petal Length 
'''

# PIPE 1st step & 2nd step, using Pipeline()
PIPE = Pipeline(
    steps=[('poly_deg2', PolynomialFeatures(degree = 2)), # 1st step
           ('linear',    LinearRegression())]             # 2nd step
    )

# we can ALREADY fit()! 
PIPE.fit(X = petal_length.reshape(-1,1),   # Petal Length
         y = iris_pd.petal_width)          # Petal Width

# prediction of Petal Width using PIPE Model
pipe_pred = PIPE.predict(X = petal_length.reshape(-1,1))

# Caluculate MSE & R-squared of PIPE model
mse_PIPE = mean_squared_error(iris_pd.petal_width, pipe_pred)
rsq_PIPE = PIPE.score(petal_length.reshape(-1,1), iris_pd.petal_width)

print(f"MSE_PIPE: {mse_PIPE.round(5)} and R-squared_PIPE: {rsq_PIPE.round(5)}")

# these are the SAME values we got when we did NOT use Pipeline!🌟

















# =============================================================================
# Modify Polynomial degree to 5
# test if same MSE and R-sq are obtained with/out PIPE
# =============================================================================

'''
without PIPE
MSE: 0.03611
Rsq: 0.93743

    
with PIPE
MSE: 0.03611
Rsq: 0.93743
'''



















"""
This is the end of "sklearn.pipeline" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""













