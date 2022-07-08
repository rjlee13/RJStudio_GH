#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 16:17:36 2022

@author: rj
"""












"""
seaborn
 (▰˘◡˘▰)


seaborn is another popular statistical graphics library!
I will show some seaborn examples in this video~

I have a cool seaborn Pair Plot example in Plots panel 👉


A lot of explanation I use in this video is from a book titled, 
"Python for Data Analysis" (Chapter 9)


Please 🌟PAUSE🌟 the video any time you want to understand code / output.
"""

# =============================================================================
# Modules needed
# =============================================================================

import seaborn as sns    # <--- FOCUS for this video

import numpy as np
import pandas as pd
from sklearn import datasets














# =============================================================================
# Histogram & Density Plot
# =============================================================================

# normal distribution with mean = 0 & sd = 1
normal_1 = np.random.normal(loc   = 0,
                            scale = 1,
                            size  = 200)

# normal distribution with mean = 13 & sd = 3
normal_2 = np.random.normal(loc   = 13,
                            scale = 3,
                            size  = 200)

# concatenate normal_1 & normal_2
normal_12 = np.concatenate([normal_1, normal_2])


# Create Histo - Density plot 
sns.distplot(a     = normal_12,
             bins  = 300,
             color = 'r')  # deprecated function  😟

# just density
sns.kdeplot(x     = normal_12,
            color = 'blue',
            fill  = True)

# just histogram
sns.histplot(x     = normal_12,
             bins  = 300,
             color = 'r')







# =============================================================================
# Prepare IRIS dataset, iris_pd
# will use iris_pd to draw different seaborn plots soon!
# =============================================================================

# get raw iris data
iris_data = datasets.load_iris()


# check feature names and data
iris_data.feature_names
iris_data.data
# create Pandas DataFrame, iris_pd
iris_pd = pd.DataFrame(data    = iris_data.data,
                       columns = iris_data.feature_names)


# check target names and target
iris_data.target_names
iris_data.target  # 0 -> setosa ; 1 -> versicolor ; 2 -> virginica
# create Pandas Categorical variable, iris_target
iris_target = pd.Categorical.from_codes(iris_data.target, 
                                        iris_data.target_names)


# add iris_target as a column called 'species' of iris_pd
iris_pd['species'] = iris_target

# check iris_pd
iris_pd # cool, this DataFrame will be used to draw more seaborn plots















# =============================================================================
# Regression Scatter Plot
# =============================================================================

# Regression Scatter Plot of "petal"
sns.regplot(x     = 'petal length (cm)',
            y     = 'petal width (cm)',
            data  = iris_pd,  # data create earlier
            ci    = 95,
            color = "red")


# Regression Scatter Plot of "sepal"
sns.regplot(x     = 'sepal length (cm)',
            y     = 'sepal width (cm)',
            data  = iris_pd,  # data create earlier
            ci    = 95,
            color = "blue")


















# =============================================================================
# Scatter Plot Matrix / Pair Plot
# =============================================================================

# Pair Plot with Histogram diagonal
sns.pairplot(data      = iris_pd,
             diag_kind = 'hist') 


# Pair Plot with kernel density estimate (kde) diagonal
sns.pairplot(data      = iris_pd,
             diag_kind = 'kde') 


# Another cool 😎 example
# 3 different colors for 3 different species
sns.pairplot(data      = iris_pd,
             kind      = 'kde',
             diag_kind = 'hist',
             hue       = 'species', # distinct colors for 3 species
             palette   = 'Dark2_r') # a specific color theme



















# =============================================================================
# Facet Grids for Categorical Variables, using sns.catplot()
# =============================================================================

# species variable in iris_pd is categorical
sns.catplot(x       = 'petal length (cm)',
            y       = 'petal width (cm)',
            hue     = 'species',
            palette = 'gnuplot_r',
            col     = 'species',  # <- 3 columns for 3 species
            data    = iris_pd)



# Let me quickly add ANOTHER categorical variable: odd_even
# every odd number row is 'odd' ; even number row is 'even'
odd_even = pd.Categorical.from_codes([0,1]*75, ['even', 'odd'])
odd_even
iris_pd['odd_even'] = odd_even
iris_pd


# Now we can ADDITIONALLY use odd_even variable 
sns.catplot(x       = 'petal length (cm)',
            y       = 'petal width (cm)',
            hue     = 'species',
            palette = 'gnuplot_r',
            col     = 'species',
            row     = 'odd_even',   # <-- odd_even
            data    = iris_pd)
# so top    row is for even category
#    bottom row is for odd  category













# =============================================================================
# Box, Swarm, Violin
# just showing different available caplot 'kind's! 😄
# =============================================================================

# kind = 'box'
sns.catplot(x       = 'species',
            y       = 'petal width (cm)',
            hue     = 'species',
            palette = 'gnuplot_r',
            kind    = 'box',
            data    = iris_pd)

# kind = 'swarm'
sns.catplot(x       = 'species',
            y       = 'petal width (cm)',
            hue     = 'species',
            palette = 'gnuplot_r',
            kind    = 'swarm',
            data    = iris_pd)

# kind = 'violin'
sns.catplot(x       = 'species',
            y       = 'petal width (cm)',
            hue     = 'species',
            palette = 'gnuplot_r',
            kind    = 'violin',
            data    = iris_pd)



















"""
This is the end of "seaborn" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""














