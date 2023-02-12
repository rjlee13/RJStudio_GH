#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:44:15 2023

@author: rj
"""










"""
Lasso Regression 
 (▰˘◡˘▰)

LASSO stands for "Least Absolute Shrinkage and Selection Operator" 🚀


Lasso regression is often used as a "Regularization technique" to prevent
overfitting to certain extent in Linear Regression. 


Lasso relies on L1-Penalty to shrink the magnitude of coefficients, 
reducing some coefficients to 0. 

Since some coefficents become 0,  
Lasso is also a "Feature Selection technique" 🤩

In Plots panel,
notice that Coefficient magnitudes are 0 when shrinkage is 10
(Code to create the visualization shown later in the video)


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""

















# =============================================================================
# Diabetes dataset
# =============================================================================
'''
Detailed information about Diabetes dataset:
    
https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset
(Link found in Description)
'''

# import sklearn's diabetes dataset & load dataset
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()  #load dataset

# check diabetes data & feature names
diabetes.data
diabetes.feature_names

# regression target, to be predicted by LASSO regression model later
diabetes.target

# =============================================================================
# Pandas DataFrame
# =============================================================================

# import pandas
import pandas as pd

# Generate Pandas DataFrame using diabetes.data & diabetes.feature_names
diabetes_pd = pd.DataFrame(data    = diabetes.data,
                           columns = diabetes.feature_names)
diabetes_pd # see our DataFrame

# Create a new column for our target
diabetes_pd['target'] = diabetes.target
diabetes_pd # it has a new column named "target"















# =============================================================================
# Lasso Regression
# =============================================================================

# import Lasso
from sklearn.linear_model import Lasso

# shrinkage, controlling Regularization strength
# as shrinkage gets BIGGER, the magnitude of coefficients shrinks to 0 FASTER
shrinkage = [0.001, 0.01, 0.1, 1, 10] # Let me try these 5 values 


# Empty DataFrame to collect Lasso Coefficients later 
coefficients_pd = pd.DataFrame(columns = diabetes.feature_names,
                               index   = shrinkage)
coefficients_pd # no data yet; all NaNs 


# Fit Lasso! (use for loop to try each shrinkage value)
for s in range(len(shrinkage)):
    
    # apply Lasso with each shrinkage value
    lasso = Lasso(alpha = shrinkage[s])
    
    # Fit Lasso; Train with our diabetes_pd data
    lasso.fit(X = diabetes_pd.loc[:, diabetes_pd.columns != 'target'],
              y = diabetes_pd['target'])
    
    # collect Lasso coefficients in coefficients_pd
    coefficients_pd.loc[shrinkage[s], :] = lasso.coef_
    
    # end of for loop

# check coefficients_pd
coefficients_pd
# coefficients values 
    # for each feature (column) 
    # at each shrinkage value (row)












# =============================================================================
# Visualize Coefficient Values
# =============================================================================

# import matplotlib
import matplotlib.pyplot as plt

# plt.semilogx: plot with log scaling on the X axis
plt.semilogx(coefficients_pd)

# details: hearders, legend, etc
plt.title('Lasso Coefficients')
plt.xlabel('Shrinkage')
plt.ylabel('Coefficient')
plt.legend(labels = coefficients_pd.columns,
           bbox_to_anchor=(1,1))
# As Shrinkage (X-axis) increases, coefficients' magnitude approaches 0 


# Another way to visualize coefficients' magnitude
plt.plot(coefficients_pd.loc[0.001], label = 'shrinkage 0.001')
plt.plot(coefficients_pd.loc[0.01],  label = 'shrinkage 0.01')
plt.plot(coefficients_pd.loc[0.1],   label = 'shrinkage 0.1')
plt.plot(coefficients_pd.loc[1],     label = 'shrinkage 1')
plt.plot(coefficients_pd.loc[10],    label = 'shrinkage 10')
plt.legend(bbox_to_anchor = (1,1))
plt.xlabel('Features')
plt.ylabel('Coefficient')
# Again, as shrinkage gets bigger, coefficients' magnitude approaches 0 
# For example, when shrinkage = 10, all coefficients are 0















"""
This is the end of "Lasso Regression" video~



Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""





















