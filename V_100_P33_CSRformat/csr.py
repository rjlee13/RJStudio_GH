#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 12:03:28 2022

@author: rj
"""
























"""
Compressed Sparse Row (CSR)
 (▰˘◡˘▰)


Lots of data are available in "Sparse Matrix" format.
Sparse Matrix's elements / values are MOSTLY 0 (zero).
And in reality, I am talking about millions of 0's in a HUGE Sparse Matrix.

Therefore, it is 🚨NOT🚨 efficient to store such data in a Matrix.
It consumes LOTS of memory!



It is much more efficient to ONLY care about NON-zero values.
One such format is called "Compressed Sparse Row (CSR)" format 🚀

This video shows you how to store Sparse Matrix into CSR format.



Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""























# =============================================================================
# Create a Model Data 
# =============================================================================
'''
The following "Model Data" will later be saved in 
1) CSR format 
2) Sparse Matrix 

In reality, Sparse Matrix is usually HUGE.
my "Model Data" is just a small example to demonstrate CSR & Sparse Matrix
'''

# Make a Model Data
model_data = {'customer_id': [3, 7, 9, 11], # 4 customers
              'product_id' : [2, 5, 6, 13], # 4 products
              'rating'     : [3, 2, 5, 1]}  # 4 customers' ratings on products
'''
this Model Data records 4 customers' ratings on 4 products
(I was thinking of Amazon 😎)
'''


# convert into a Pandas DataFrame
import pandas as pd
model_data_pd = pd.DataFrame(data = model_data)

# check our Model Data
model_data_pd





















# =============================================================================
# Numpy Array
# =============================================================================
'''
Numpy Arrays (np.array) work smoothly with functions I need for 
creating Compressed Sparse Row (CSR) format 

So, let me quickly convert our data into np.array
'''

import numpy as np

# Ratings in NumPy array
data_np = np.array(model_data_pd['rating'])
data_np

# Customers in NumPy array
row_index = np.array(model_data_pd['customer_id'])
row_index

# Products in NumPy array
col_index = np.array(model_data_pd['product_id'])
col_index





















# =============================================================================
# CSR Matrix Format 🔥
# =============================================================================

# create Compressed Sparse Row (CSR) format
# data_np, row_index, col_index were all created just earlier
from scipy.sparse import csr_matrix
csr = csr_matrix((data_np, (row_index, col_index)),
                 dtype = int)

# check CSR format 
print(csr)


# we can retrieve a particular rating like this:
csr[3,2] # 3, rating given by customer_id==3 & product_id==2 


# If we give a ⛔NON-existing⛔ [customer_id, product_id] pair,
csr[2,2] # it returns 0
csr[5,7] # 0


# Convert CSR format to Sparse Matrix, using toarray()
csr_sparse_matrix = csr.toarray()
csr_sparse_matrix
'''
Lots of 0's are placed when NO rating data is available

Again, in reality, a sparse matrix can be much bigger with much more 0's
'''




















# =============================================================================
# Comparing 3 Data Formats
# =============================================================================
'''
Let me put together
    1) Pandas Model Data
    2) CSR format
    3) Sparse matrix 

Just so that it is easier to see the difference     
'''

model_data_pd
print(csr)
csr_sparse_matrix
























"""
This is the end of "CSR" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""



















