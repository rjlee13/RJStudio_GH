#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 16:01:08 2022

@author: rj
"""

import numpy as np













"""
StandardScaler & OneHotEncoder
 (▰˘◡˘▰)


This short video introduces 2 very commonly used data preprocessors
from sklearn:
    1) StandardScaler
    2) OneHotEncoder

If you are studying Data Science related topics, you are very likely to
encounter those 2 preprocessors 🚀

Without further ado, let me show you! 🔥


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""
# =============================================================================
# Import
# =============================================================================

# For IRIS dataset
from sklearn import datasets

# For StandardScaler & OneHotEncoder (from scikit-learn)
from sklearn import preprocessing
















# =============================================================================
# StandardScaler 
# =============================================================================

# Load IRIS data 🍀
iris_data = datasets.load_iris() 

# only use 2 features
iris_x = iris_data.data[:, :2]
iris_x # we see 2 columns / features




# Standardize!  First, create standard scaler ✨
std_scaler = preprocessing.StandardScaler()


# fit our IRIS data; it calculates feature means!
std_scaler.fit(iris_x).mean_ # [5.84333333, 3.05733333]
# check Numpy also gives the same feature means
np.mean(iris_x, axis=0)      # [5.84333333, 3.05733333], same means 


# fit our IRIS data; it calculates feature variances!
std_scaler.fit(iris_x).var_ # [0.68112222, 0.18871289]
# check Numpy also gives the same feature variances
np.var(iris_x, axis = 0)    # [0.68112222, 0.18871289], same variances



# standardize IRIS data, using transform()
std_scaler.transform(iris_x)

# fit() & transform() can be conducted at once, using fit_transform()
std_scaler.fit_transform(iris_x)


# we can standardize NEW data, for example [[-2, 5]]
std_scaler.transform([[-2, 5]])  # [-9.50359997,  4.47195585]

# MANUALLY calculate standardized values to check
# using mean & variance values obtained from above
(-2-5.84333333) / np.sqrt(0.68112222)  # -9.5035
(5-3.05733333) / np.sqrt(0.18871289)   # 4.47195















# =============================================================================
# OneHotEncoder
# =============================================================================

# iris target has 3 unique values specifying 3 species 
iris_data.target_names # ['setosa', 'versicolor', 'virginica']

# species data as iris_y 🍀
iris_y = iris_data.target.reshape(-1,1)
iris_y
np.unique(iris_y, return_counts = True)
# 3 unique values for 3 species: [0, 1, 2]
# and 50 species each



# OneHotEncoder! First, create OHEncoder ✨
OHencode = preprocessing.OneHotEncoder(sparse = False)
# fit our IRIS species data
OHencode.fit(iris_y)
# 3 species are already recorded as 0, 1, 2
OHencode.categories_  # [0, 1, 2]


# One Hot Encoding, using transform()
OHencode.transform(iris_y)
# first  50 rows: left   column is 1
# second 50 rows: middle column is 1
# third  50 rows: right  column is 1

# So, each column represents a specie

# Again, fit() & transform() can be conducted at once, using fit_transform()
OHencode.fit_transform(iris_y)





# By DEFAULT, `sparse = True`  ;  NOTE I put `sparse = False` above
OHencode_sp = preprocessing.OneHotEncoder(sparse = True)

# When `sparse = True`, result is saved in CSR format! 🚨
OHencode_sp.fit_transform(iris_y) # stored elements in Compressed Sparse Row
'''
I made a YouTube video explaining CSR format some time ago.
If you are interested, YouTube link is in Description
'''

# CSR format can be viewed with print()
print(OHencode_sp.fit_transform(iris_y))

# If you want to see OHE result (like above), use toarray()
OHencode_sp.fit_transform(iris_y).toarray()
# again we see
        # first  50 rows: left   column is 1
        # second 50 rows: middle column is 1
        # third  50 rows: right  column is 1















"""
This is the end of "StandardScaler & OneHotEncoder" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""



















