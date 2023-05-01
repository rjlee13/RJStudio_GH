#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 15:18:47 2023

@author: rj
"""









"""
One-way ANOVA
 (▰˘◡˘▰)


One-way ANOVA is a frequently used Statistical method to compare 
the MEANS of 2 or more independent groups! 🚀



Before conducting 1-way ANOVA, we check the followings:
    1) groups have normal population distributions
    2) groups have the same variances
    
For Normality check, we conduct Shapiro-Wilk test.
For variance check,  we conduct Levene test.



In famous IRIS 💐 dataset, there exist 3 different species.
We will test if the 3 species' MEAN Sepal Widths are 
statistically different to one another.


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""

















# =============================================================================
# IRIS 💐 dataset
# =============================================================================

# load IRIS data
from sklearn.datasets import load_iris
iris = load_iris()


# create Pandas DataFrame using IRIS 💐 dataset
import pandas as pd
iris_pd = pd.DataFrame(data = iris.data,
                       columns = ['sepal_length', 'sepal_width', 
                                  'petal_length', 'petal_width'])
iris_pd['species'] = iris.target
iris_pd['species'] = iris_pd['species'].map({0:'setosa', 
                                             1:'versicolor', 
                                             2:'virginica'})
# check Pandas DataFrame
iris_pd     


# Quick Visualization
import seaborn as sns
sns.scatterplot(data   = iris_pd,
                x      = 'species',
                y      = 'sepal_width',
                s      = 100,
                style  = 'species',
                legend = False)
'''
As you can see from the scatterplot 👀, 

it seems like MEAN sepal widths for the 3 species are different!
Setosa's sepal width seems to be the longest
Versicolor shortest

But our goal is to come up with statistical evidence that they are different
'''
















# =============================================================================
# Assumptions  
# 1) Normal Population Distribution 
# 2) Same Variance
# =============================================================================

# create 'sepal-width' data (Pandas Series) for each specie
setosa_sw     = iris_pd.sepal_width[iris_pd.species == 'setosa'] 
versicolor_sw = iris_pd.sepal_width[iris_pd.species == 'versicolor']
virginica_sw  = iris_pd.sepal_width[iris_pd.species == 'virginica']



# 1) Normal Population Distribution check - Shapiro-Wilk test
from scipy.stats import shapiro

shapiro(setosa_sw).pvalue      # pvalue=0.27..
shapiro(versicolor_sw).pvalue  # pvalue=0.33..
shapiro(virginica_sw).pvalue   # pvalue=0.18..

# All the pvalues are greater than 0.05 threshold,
# so the test did NOT show evidence of non-normality.
# For simplicity, let's 'assume' Normality is satisfied here.



# 2) Same Variance check - Levene test
from scipy.stats import levene

levene(setosa_sw, versicolor_sw, virginica_sw).pvalue # pvalue=0.55..

# pvalue is greater than 0.05 threshold.
# Again, for simplicity, we assume that the 3 species' sepal widths
# are from populations with equal variances.














# =============================================================================
# One-way ANOVA test (plus Post Hoc Analysis)
# =============================================================================

# One-way ANOVA
from scipy.stats import f_oneway

f_oneway(setosa_sw, versicolor_sw, virginica_sw).pvalue # 4.49..e-17

# pvalue is essentially 0
# Therefore, we REJECT the Null Hypothesis that
# the 3 species have the same sepal width means!

'''
One-way ANOVA tells us at least one specie's mean is different from the rest

We want to know the SPECIFIC differences in MEAN sepal widths of 3 species!

Then, we must conduct Post-Hoc analysis/test~
'''

# Post-Hoc Analysis: Pairwise (Multi-Comparison) Difference
from statsmodels.stats.multicomp import MultiComparison

# Conduct Tukey's Honest Significant Difference (HSD) test
mc = MultiComparison(data   = iris_pd.sepal_width, 
                     groups = iris_pd.species)
tukeyhsd = mc.tukeyhsd(alpha = 0.05)

fig = tukeyhsd.plot_simultaneous() # visualize
print(tukeyhsd.summary())          # detail

'''
Plot and the summary detail show us pairwise difference among 3 species.

The plot shows setosa > virginica > versicolor

Summary detail shows adjusted-pvalues are all smaller than 0.05.
So the 3 species' MEAN sepal widths are all statistically different 
to one another.
'''


























"""
This is the end of "1-way ANOVA" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""





















