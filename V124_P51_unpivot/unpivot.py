#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 15:09:41 2023

@author: rj
"""

















"""
Unpivot Pandas DataFrame
 (▰˘◡˘▰)


In this video, I demonstrate how to "unpivot" Pandas DataFrame.
Unpivot means to change the format of the data from "WIDE" to "LONG". 🚀


The following page has a nice 'unpivot' visualization:
    https://learn.microsoft.com/en-us/power-query/unpivot-column


Then I perform simple data visualization using the Unpivoted DataFrame!

Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""


















# =============================================================================
# Unpivot [Wide -> Long]
# =============================================================================

# import pandas
import pandas as pd

# dictionary data
data_dictionary = {'name'       : ['Michael','Jack', 'Sally'],
                   'age'        : [27, 39, 19],
                   'weight(kg)' : [71, 65, 49],
                   'height(cm)' : [182, 170, 162]}

# generate Pandas DataFrame using above dictionary
data_pandas = pd.DataFrame(data = data_dictionary)
data_pandas # WIDE format


# Unpivot! 
# use pd.melt() 
data_unpivot = data_pandas.melt(
    id_vars    = 'name',
    var_name   = 'attribute',
    value_name = 'value')

data_unpivot # LONG format
# NOTICE 'age', 'weight(kg)', 'height(cm)' are now in "attribute" column

# 'age', 'weight(kg)', 'height(cm)' used to be separate columns in WIDE format











# =============================================================================
# Seaborn Visualization
# =============================================================================
'''
Now that we have Unpivot data, we can have 'age', 'weight(kg)', 'height(cm)' 
together in X-axis
'''

# import seaborn
import seaborn as sns

# scatterplot
sns.scatterplot(data   = data_unpivot,# Unpivot data
                x      = 'attribute', # 'age', 'weight(kg)', 'height(cm)'
                y      = 'value',
                hue    = 'name',
                legend = False)

# we could draw a lineplot on top of scatterplot
sns.lineplot(data   = data_unpivot,   # Unpivot data
                x   = 'attribute',    # 'age', 'weight(kg)', 'height(cm)'
                y   = 'value',
                hue = 'name')

















# =============================================================================
# Another quick example
# =============================================================================

# Suppose we have the following "precipitation" data
# I made up the numbers
year_data = {'2000' : 35,
             '2001' : 31,
             '2002' : 39,
             '2003' : 29,
             '2004' : 41,
             '2005' : 42,
             '2006' : 33,
             '2007' : 26}

# create pandas DataFrame
year_data_pd = pd.DataFrame(data  = year_data,
                            index = [0])
year_data_pd # WIDE format

# Unpivot
year_data_unpivot = year_data_pd.melt(
    var_name   = 'year',
    value_name = 'precipitaion')

year_data_unpivot # check LONG format after unpivot

# now we can have all the years in the X-axis

# scatterplot
sns.scatterplot(data   = year_data_unpivot,
                x      = 'year',         # 2000 ~ 2007
                y      = 'precipitaion')

# draw a lineplot on top 
sns.lineplot(data   = year_data_unpivot,
                x   = 'year',            # 2000 ~ 2007
                y   = 'precipitaion')





















"""
This is the end of "Unpivot Pandas DataFrame" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""



















