#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:50:11 2022

@author: rj
"""



"""

 (▰˘◡˘▰)


In this video, I show how


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
# 
# =============================================================================


df1 = pd.DataFrame({'key': ['A', 'B', 'C'] * 4,
                    'value': np.arange(12)})


df1[df1.key == 'A'].value.mean()
df1[df1.key == 'B'].value.mean()
df1[df1.key == 'C'].value.mean()


df1.groupby(by = 'key').agg(func = 'mean')
df1.groupby(by = 'key').size()









# =============================================================================
# 
# =============================================================================

df2 = pd.DataFrame({'data1': np.arange(6),
                    'data2': np.arange(6)*10,
                    'key1' : ['a','b','a','b','a','b'],
                    'key2' : ['c','c','c','d','d','d']})
df2

df2.groupby(by = 'key1').mean()
df2.groupby(by = ['key1', 'key2']).mean()


# select specific data
df2.groupby(by = 'key1').mean()
df2.groupby(by = 'key1').data1.mean()
df2.groupby(by = 'key1').data2.mean()





# =============================================================================
# GroupBy object
# =============================================================================

df2
df2_group_key2 = df2.groupby(by = 'key2')
df2_group_key2.mean()
df2_group_key2.describe()




# =============================================================================
# Data Aggregation
# =============================================================================

df2
df2.groupby(by = 'key2').mean()
df2.groupby(by = 'key2').count()
df2.groupby(by = 'key2').agg(['count','mean'])





# =============================================================================
# 
# =============================================================================


grouped_key = df1.groupby(by = 'key').value
grouped_key
grouped_key.mean()



grouped_key.transform(lambda x: x.mean())
# built-in function, like "mean"
grouped_key.transform('mean')


grouped_key.transform(lambda x: 10*x)



# =============================================================================
# 
# =============================================================================
















"""
This is the end of "Categorical Data" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""








