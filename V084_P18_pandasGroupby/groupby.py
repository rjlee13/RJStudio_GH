#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:50:11 2022

@author: rj
"""










"""
GroupBy
 (▰˘◡˘▰)


GroupBy is a very important technique 
especially for conducting EDA, Exploratory Data Analysis. 


I show several GroupBy examples in this video! 🚀


A lot of explanation I use in this video is from a book titled, 
"Python for Data Analysis" (Chapters 10 & 12)


Please 🌟PAUSE🌟 the video any time you want to understand code / output.
"""

# =============================================================================
# Modules needed
# =============================================================================

import pandas as pd
import numpy as np
















# =============================================================================
# A Simple (motivating) Example
# =============================================================================

# consider following simple data
df1 = pd.DataFrame({'key': ['A', 'B', 'C'] * 4,
                    'value': np.arange(12)})
df1


# Let me get mean() value for each each key: A B C
df1[df1.key == 'A'].value.mean() # 4.5
df1[df1.key == 'B'].value.mean() # 5.5
df1[df1.key == 'C'].value.mean() # 6.5
# (please PAUSE and make sure you understand the output by examining df1)



# I can find each key's mean value like this ✨ at ONCE
df1.groupby(by = 'key').mean()
# OR like this
df1.groupby(by = 'key').agg(func = 'mean')


# size() is also pretty useful method to know :)
df1.groupby(by = 'key').size() # 4 for each key
# because df1 contains 4 values for each key















# =============================================================================
# Multiple Keys 🔑🔑 (and multiple data)
# GroupBy works with multiple keys too!
# =============================================================================

# consider following data with multiple keys & data
df2 = pd.DataFrame({'data1': np.arange(6),
                    'data2': np.arange(6)*10,
                    'key1' : ['a','b','a','b','a','b'],  # FIRST  key 🔑
                    'key2' : ['c','c','c','d','d','d']}) # SECOND key 🔑
df2


# GroupBy just one key1 
df2.groupby(by = 'key1').mean()
# and we get mean value for each key1: a & b
# for BOTH data1 & data2


# This time,
# GroupBy BOTH key1 & key2 🔑🔑
df2
df2.groupby(by = ['key1', 'key2']).mean()
df2.groupby(by = ['key1', 'key2']).size()
# please PAUSE and understand how the outputs are computed by examining df2


# select specific data (data1 or data2)
df2.groupby(by = 'key1').mean()        # output for BOTH data1 & data2
df2.groupby(by = 'key1').data1.mean()  # just for data1 
df2.groupby(by = 'key1').data2.mean()  # just for data2


















# =============================================================================
# GroupBy object
# =============================================================================

# consider df2 from above again
df2

# let me create a GroupBy object, df2_group_key2
df2_group_key2 = df2.groupby(by = 'key2')
df2_group_key2 # GroupBy object, has NOT computed anything just yet!


# find mean() for each key2 using the GroupBy object
df2_group_key2.mean()

# so above is the SAME as:
df2.groupby(by = 'key2').mean()



# we can describe() the GroupBy object 🌟
df2_group_key2.describe()
# we get whole bunch of useful information!













# =============================================================================
# Data Aggregation
# =============================================================================

# consider df1 from earlier 
df1

# we saw GroupBy result using mean() & size() earlier already
df1.groupby(by = 'key').mean()
df1.groupby(by = 'key').size()


# we can calculate BOTH at the same time using agg() like this:
df1.groupby(by = 'key').agg(func = ['size','mean'])



# we can even use our OWN functions 🔥
# let me create a function called range: basically, max - min
range = lambda group: group.max() - group.min() 
# test
range(pd.Series([3,5,10])) # 7 = 10 - 3, good 

# let's use range function inside agg()
df1
df1.groupby(by = 'key').agg(range)
# 9 for each key, good!
    # A's smallest value is 0, largest is 9.  So,  9 - 0 = 9
    # B's smallest value is 1, largest is 10. So, 10 - 1 = 9


# of course, range can be used with other methods such as 'mean'
df1.groupby(by = 'key').agg(['mean', range])














# =============================================================================
# Custom Column Header/Name
# =============================================================================

# consider df1 from earlier 
df1


# Let me compute GroupBy mean 
df1.groupby(by = 'key').agg('mean')
# Output alone does NOT show 'what' was computed


# we can put a custom column header using tuple like this:
df1.groupby(by = 'key').agg([('Average','mean')])
# now we see custom name: 'Average'


















# =============================================================================
# Multiple Functions to Multiple Data (at once) 🤩
# we can use Dictionary
# =============================================================================

# consider df2 from earlier 
df2


# min for data1 ; max for data2 AFTER grouping by key2
df2.groupby(by = 'key2').agg({'data1': 'min',
                              'data2': 'max'})


# min & max for data1 ; mean for data2 AFTER grouping by key2
df2.groupby(by = 'key2').agg({'data1': ['min', 'max'],
                              'data2': 'mean'})



# (please PAUSE and make sure you understand the above outputs ^.^ )
















# =============================================================================
# transform() 
# =============================================================================

# consider df3
df3 = pd.DataFrame({'key': ['A', 'A', 'A', 'B', 'B', 'B'] ,
                    'value': [1,1,1, 7,8,9]})
df3

# let's create a GroupBy object
df3_group_key = df3.groupby(by = 'key')
# quick check if mean() works
df3_group_key.mean()


# Suppose I want to replace each value by key-grouped mean
# then we can use transform() like this:
df3_group_key.transform('mean') 
# notice all A key values are replaced to 1.0, key-A mean
#        all B key values are replaced to 8.0, key-B mean


# of course, you can use your custom functions
# remember range from earlier:
    """
    range = lambda group: group.max() - group.min()
    """
df3    
df3_group_key.transform(range)
# notice all A key values are replaced to 0 (= 1 - 1)
#        all B key values are replaced to 2 (= 9 - 7)

























"""
This is the end of "GroupBy" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""















