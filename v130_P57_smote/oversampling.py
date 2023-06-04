#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 15:16:18 2023

@author: rj
"""













"""
SMOTE: Synthetic Minority Over-sampling Technique
 (▰˘◡˘▰)



In this video, I perform SMOTE, which is a commonly used 
"OverSampling Technique" to overcome class imbalance in a dataset. 🚀

To be more specific, it addresses class imbalance problem by 
increasing the "minority" class.



Let's read about SMOTE algorithm by asking ChatGPT~! 🦾



Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""


















# =============================================================================
# create IMbalanced Dataset
# =============================================================================

# import make_classification
from sklearn.datasets import make_classification

# create IMbalanced dataset with 2000 samples & 2 classes/categories
x, y = make_classification(
    n_samples    = 2000,         # 2000 samples
    n_classes    = 2,            # 2 classes/categories 
    weights      = [0.95, 0.05], # majority 95% ; minority 5% <- IMbalanced!
    n_features   = 5,            # 5 features/columns
    flip_y       = 0,            # set 0 for NO random assignments
    random_state = 2             # for reproducibility
    )          


# import Pandas
import pandas as pd

# create Pandas DataFrame using x
x_pd = pd.DataFrame(
    data    = x,
    columns = ['1st', '2nd', '3rd', '4th', '5th'] # 5 features/columns
    )
# create Pandas Series using y
y_pd = pd.Series(
    data = y, 
    name = 'label')

# check x_pd & y_pd
x_pd # notice 5 features/columns
y_pd # Pandas Series, let's check its distribution:
y_pd.value_counts() # majority 1900 samples ; minority 100 samples
                    # majority 95% ; minority 5%    <-- IMbalanced ✅












# =============================================================================
# Ramdon OverSampling  (ROS)
# =============================================================================

# import RandomOverSampler
from imblearn.over_sampling import RandomOverSampler

# Conduct Random OverSampling, ros
ros = RandomOverSampler(sampling_strategy = 'minority')
x_ros, y_ros = ros.fit_resample(x_pd, y_pd)

# check distribution
y_ros.value_counts() # NOW both classes 1900 samples
                     # successfully oversampled minority class

# =============================================================================
# this time... SMOTE! 🌟
# =============================================================================

# import SMOTE
from imblearn.over_sampling import SMOTE

# Conduct SMOTE
smote = SMOTE(sampling_strategy = 'minority')
x_smote, y_smote = smote.fit_resample(x_pd, y_pd)

# check distribution
y_smote.value_counts() # AGAIN, both classes 1900 samples
                       # successfully oversampled minority class


'''
To clearly see the difference between ROS & SMOTE

let's VISUALIZE the 2 oversampled results!
'''











# =============================================================================
# Visualization: SMOTE vs ROS 
# =============================================================================

# import matplotlib & seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# prepare to visualize 3 plots together 
fig, axes = plt.subplots(ncols = 3)

# original dataset <- NO OverSampling (left)
sns.scatterplot(x = x_pd['1st'], 
                y = x_pd['2nd'],
                hue = y_pd,
                alpha = 0.5,
                ax = axes[0])
axes[0].set_title('Original')

# Random OverSampling (middle)
sns.scatterplot(x = x_ros['1st'], 
                y = x_ros['2nd'],
                hue = y_ros,
                alpha = 0.5,
                ax = axes[1])
axes[1].set_title('ROS')

# SMOTE (right)
sns.scatterplot(x = x_smote['1st'], 
                y = x_smote['2nd'],
                hue = y_smote,
                alpha = 0.5,
                ax = axes[2])
axes[2].set_title('SMOTE')

'''
ROS just increases number of minority samples
at the SAME places where original minority samples existed.

SMOTE, on the other hand, generates new minority classes 
at DIFFERENT locations, thereby increasing "variety".
'''


















"""
This is the end of "SMOTE" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""





















