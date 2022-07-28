#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 19:58:18 2022

@author: rj
"""



# =============================================================================
# Sparse Matrix practice ch8.1
# =============================================================================

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix # FOCUS for this chapter


# sample small data
ratings = {'user_id'  : [1,2,4],
           'movie_id' : [2,3,7],
           'rating'   : [4,3,1]}

# pandas dataframe
ratings = pd.DataFrame(ratings)
ratings

# using Pandas Pivot to make full matrix
rating_matrix = ratings.pivot(index   = 'user_id',
                              columns = 'movie_id',
                              values  = 'rating').fillna(0)
rating_matrix

# making into numpy array
full_matrix1= np.array(rating_matrix)
full_matrix1



# creating sparse matrix
data = np.array(ratings['rating'])
data

row_indices = np.array(ratings['user_id'])
row_indices

col_indices = np.array(ratings['movie_id'])
col_indices

rating_matrix_CSR = csr_matrix((data, (row_indices,col_indices)),
                               dtype = int)
rating_matrix_CSR
print(rating_matrix_CSR) # sparse matrix!


# full matrix
# compare the result below against full_matrix1
full_matrix2 = rating_matrix_CSR.toarray()
full_matrix2
full_matrix2[1,2] # pick rating 4 element















