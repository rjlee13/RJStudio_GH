#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 23:27:59 2022

@author: rj
"""











"""
Categorical Data
 (▰˘◡˘▰)


In this video, I show how Pandas package handles Categorical data!


A lot of explanation I use in this video is from a book titled, 
"Python for Data Analysis" (Chapter 12)


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""

# =============================================================================
# Modules needed
# =============================================================================

import pandas as pd
import numpy as np
















# =============================================================================
# Categorical Representation
# =============================================================================

# suppose we have the following pandas Series
values = pd.Series([0,1,0,0]*2)
values

# we can get unique values and their counts like this
pd.unique(values)
pd.value_counts(values)
# OR using Numpy
np.unique(values, return_counts = True)


# suppose I want 0 to represent apple; 1 orange
dim = pd.Series(['apple','orange'])
dim

# then take() method can "categorize" 2 fruits
# using pandas Series 'values' from above
dim.take(values)
dim.take(values).unique()
dim.take(values).value_counts()














# =============================================================================
# Categorical Type
# =============================================================================

# consider simple fruit data:
fruits = ['apple', 'orange', 'apple', 'apple'] * 2
N = len(fruits)
df = pd.DataFrame({'fruit': fruits,
                   'id': np.arange(N)})
df

# right now, fruit column is Python string 'objects'
df.dtypes # object; NOT categorical


# CONVERT 🔥 fruit column to categorical, using .astype('category')
df['fruit'] = df['fruit'].astype('category') 

# NOW we see 'category' for fruit column
df.dtypes      # category
df.fruit.dtype # CategoricalDtype(categories=['apple', 'orange'], ordered=False)
df['fruit'].dtype # same as above















# =============================================================================
# pandas.Categorical
# =============================================================================

# use pd.Categorical()
my_category = pd.Categorical(['A', 'B', 'B', 'C', 'A'])
my_category # Categories (3, object): ['A', 'B', 'C']


# OR 
# 1. you can create distinct categories first
distinct_category = ['A', 'B', 'C']

# 2. encode them with integer key
encode = [0, 1, 1, 2, 0]           # 0 -> A ; 1 -> B ; 2 -> C

# 3. now use from_codes()
pd.Categorical.from_codes(encode, distinct_category)
# ['A', 'B', 'B', 'C', 'A'] 
# same result as above
















# =============================================================================
# Ordering of the Categories
# =============================================================================

# Suppose A, B, C were school grades
grade_category = ['A', 'B', 'C']
encode = [0, 1, 1, 2, 0] # 0 -> A ; 1 -> B ; 2 -> C


# now notice `ordered = True`
grades = pd.Categorical.from_codes(encode, distinct_category,
                                   ordered = True)
grades         # ['A' < 'B' < 'C'] <- now we have an order


# BUT that looks like C is better grade than B, and B better than A...😅
# we can RE-order, using reorder_categories()
grades_reorder = (grades
                  .reorder_categories(['C', 'B', 'A'], ordered = True))
grades_reorder # ['C' < 'B' < 'A']















# =============================================================================
# pd.qcut()
# =============================================================================

# generate random 1000 integers between 0 and 99
np.random.seed(7)
rand_int = np.random.randint(100, size = 1000)
rand_int.min() # 0
rand_int.max() # 99


# generate quartile categories, using pd.qcut()
quartile = pd.qcut(rand_int, 4)
quartile.dtype # we got ordered Categorical type:
# categories=[(-0.001, 25.0], (25.0, 49.0], (49.0, 74.25], (74.25, 99.0]],
# ordered=True


# INSTEAD of getting the exact quartile range, 
# it may be convenient to use: ['Q1', 'Q2', 'Q3', 'Q4'] label
quartile = pd.qcut(rand_int, 4,
                   labels = ['Q1', 'Q2', 'Q3', 'Q4'])
quartile.dtype # now we see
# categories=['Q1', 'Q2', 'Q3', 'Q4'], ordered=True)


# for quick frequency check of each quartile, you can do:
pd.crosstab(quartile, columns = 'count')

# OR even better (using groupby & agg) 
pd.Series(rand_int).groupby(quartile).agg(['count', 'min', 'max'])














# =============================================================================
# Smaller Memory Usage
# =============================================================================

# 10 million
tenM = 10000000

# when it's not category
bad = pd.Series(['A','B','C','D'] * (tenM//4)) 
bad # Length: 10000000, dtype: object


# when we change to category
good = bad.astype('category')
good # Length: 10000000, dtype: category


# Check memory usage difference! 🔥
bad.memory_usage()  # 80000128
good.memory_usage() # 10000332

# 8-fold difference!! 🫢
















# =============================================================================
# Categorical Methods
# =============================================================================

# nice pandas series
nice = pd.Series(['A','B','C','D'] * 2)

# make it into category type, using astype()
nice_cat = nice.astype('category')
nice_cat # Categories (4, object): ['A', 'B', 'C', 'D']

# cat.codes & cat.categories
nice_cat.cat.codes       # A -> 0 , ... , D -> 3
nice_cat.cat.categories  # ['A', 'B', 'C', 'D']


#----


# let's say that there is SUPPOSED to be an 'E' category 
# even though there is NO data belonging to 'E' category
nice_cat_E = nice_cat.cat.set_categories(['A','B','C','D','E'])
nice_cat_E # we see Categories (5, object): ['A', 'B', 'C', 'D', 'E']
           # even though we do NOT have any E data

# notice the difference
nice_cat.value_counts()   # NO 'E' category
nice_cat_E.value_counts() # 'E' category shows up with 0 count!


#----


# let's say I ONLY want categories 'A' & 'B'
nice_cat_AB = nice_cat[nice_cat.isin(['A','B'])]
nice_cat_AB # notice there STILL are 'C' & 'D' categories

# to completely DISCARD 'C' & 'D' categories
# use cat.remove_unused_categories()
nice_cat_AB.cat.remove_unused_categories()
# now 'C' & 'D' categories gone!
















# =============================================================================
# One-hot encoding
# categorical variable -> dummy variable
# =============================================================================

'''
"this technique involves creating a DataFrame
with a column for each distinct category" - from textbook

Also, this technique is frequently used in Deep Learning algorithms!
'''

# suppose we have some school grade category
grade_cat = pd.Series(['A', 'B', 'C', 'D'] * 2,
                      dtype = 'category')
grade_cat # Categories (4, object): ['A', 'B', 'C', 'D']


# One-hot encode
pd.get_dummies(grade_cat)




















"""
This is the end of "Categorical Data" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""

















