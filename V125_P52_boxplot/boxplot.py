#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 23:57:02 2023

@author: rj
"""














"""
Boxplot
 (▰˘◡˘▰)


In this video, I show how to create boxplots in Python.
Boxplots are very commonly used to show the spread of numerical data. 🚀

I have an example on the right 👉


And I create a function to identify "outliers", which are 
indicated as small circles found at the top and bottom of boxplots.
( I assume you know how to interpret/read boxplots! )



Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""




















# =============================================================================
# Iris Data 🌿 (Data Processing)
# =============================================================================

# import pandas & iris dataset
import pandas as pd
from sklearn.datasets import load_iris

# Load iris Data
iris_raw = load_iris()

# data & column names from iris_raw
iris_raw.data
iris_raw.feature_names

# create Pandas DataFrame using data & column names 
iris_pd = pd.DataFrame(iris_raw.data, 
                       columns= iris_raw.feature_names)
iris_pd


# species are stored as target in iris_raw
iris_raw.target 
# 0 is setosa ; 1 is versicolor ; 2 is virginica

# create a new 'species' column in iris_pd and store target
iris_pd['species'] = iris_raw.target
iris_pd # notice new column 'species' contains iris_raw.target data


# create 'species_label' column in iris_pd
species_label = {0:'setosa',1:'versicolor',2:'virginica'}
iris_pd['species_label'] = iris_pd['species'].map(species_label)
iris_pd 
# notice new column 'species_label' shows the name of species
    # Ex) 0 is setosa ; 2 is virginica
















# =============================================================================
# Boxplots 🌟
# =============================================================================

# import matplotlib
import matplotlib.pyplot as plt


# since boxplots show spread of NUMERICAL data, 
# we need to "drop" NON-numerical columns: species & species_label
plt.boxplot(iris_pd.drop(columns = ['species','species_label']))

# hmm... would be nice to show column names along the X-axis

# Show column names along the X-axis, using plt.xticks()
plt.xticks([1, 2, 3, 4], 
           iris_pd.drop(columns = ['species','species_label']).columns)




# Pandas DataFrame has boxplot method 
iris_pd[['sepal width (cm)']].boxplot()

# Generate boxplot for each of the 3 species using "by = spcies_label"
iris_pd[['sepal width (cm)', 'species_label']].boxplot(by = 'species_label')
# notice different species are separated!

# title looks messy; let me eliminate "sepal width (cm)"
plt.title("")




















# =============================================================================
# Detect Outliers  🚔🚨
# =============================================================================

# Let me quickly draw Boxplot of sepal width
plt.boxplot(iris_pd['sepal width (cm)'], whis = 1.5)
plt.title("Boxplot")
plt.xticks([1], ['sepal width (cm)'])

'''
Boxplot on the right tells us that there are 4 outliers:
    3 at the top
    1 at the bottom
    
Let me write a function for detecting the 4 outliers!
'''

# import numpy
import numpy as np

# function
def detect_outlier(data, column):
    
    # 25 percentile; q1  &  75 percentile; q3
    q1 = np.percentile(data[column], 25)
    q3 = np.percentile(data[column], 75)
    
    # calculate IQR
    iqr = q3 - q1
    
    # upper & lower limit
    upper_limit = q3 + 1.5 * iqr # outlier, if GREATER than upper_limit
    lower_limit = q1 - 1.5 * iqr # outlier, if SMALLER than lower_limit
    
    # Outliers
    # samples GREATER than upper_limit OR SMALLER than lower_limit
    outlier = data[(data[column] > upper_limit) | (data[column] < lower_limit)]
    return(outlier[column])


# detect the 4 outliers!
detect_outlier(iris_pd, 'sepal width (cm)')
# 4.4, 4.2, 4.1 are the 3 outliers at the top
# 2.0 is the outlier at the bottom

















"""
This is the end of "Boxplot" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""






















