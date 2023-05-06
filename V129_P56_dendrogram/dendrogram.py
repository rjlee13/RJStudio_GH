#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:15:33 2023

@author: rj
"""









"""
Dendrogram, Hierarchical Clustering
 (▰˘◡˘▰)


Hierarchical Clustering (HC) does NOT require us to pre-specify 
the number of clusters (unlike K-Means clustering). 
In fact, HC can be used to obtain any number of clusters. 



Also, HC generates a nice tree-based representation of all observations,
called Dendrogram! 🌲
I have an example in the Plots Panel on the right 👉



In this video, I perform HC to divide 50 US 🇺🇸 States into 
a reasonable number of clusters, using US crime dataset! 


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""



















# =============================================================================
# USArrests Dataset
# =============================================================================

# import pandas and load USArrests dataset
import pandas as pd
usarrests = pd.read_csv('./Desktop/RJstudio/V129/USArrests.csv')
'''
I got USArrests.csv from
https://github.com/JWarmenhoven/ISLR-python/blob/master/Notebooks/Data/USArrests.csv
(Link in Description)
'''

# notice that first column name is "Unnamed: 0"  <- should change to "State"
usarrests.head()
usarrests.columns

# change first column name to "State"
usarrests = usarrests.rename(columns={'Unnamed: 0':'State'})
usarrests.head()
usarrests.columns # changed to "State", good!

# check dimension of dataset
usarrests.shape # 50 States with 5 features 

'''
Feature Explanation below

State:    50 US States
Murder:   Murder arrests (per 100,000)
Assault:  Assault arrests (per 100,000)
UrbanPop: Percent urban population
Rape:     Rape arrests (per 100,000)

(Data recorded in 1973)
'''













# =============================================================================
# Linkage - defines dissimilarity between 2 groups
# =============================================================================

'''
5 common types of linkage are

1) single   : Minimum distance between 2 clusters
2) complete : Maximum distance between 2 clusters
3) average  : Average distance between 2 clusters
4) centroid : distance between 2 clusters' centroids
5) ward**   : analyzes the variance of clusters

**Ward linkage is sometimes described as the most suitable linkage to use
for quantitative variables. 
I encourage you to read more about ward linkage!


The result of Hierarchical Clustering depends largely on the 
type of linkage used. 🔥
''' 




















# =============================================================================
# Single Linkage Dendrogram
# =============================================================================

# import dendrogram & linkage
from scipy.cluster.hierarchy import dendrogram, linkage

# single linkage
single_link = linkage(
    y      = usarrests.drop(['State'], axis = 1), # exclude State column
    method = 'single')

# generate dendrogram using the single linkage above
dendrogram(Z               = single_link,
           labels          = usarrests.State.tolist(),
           color_threshold = 25)

# good to draw horizontal line corresponding to `color_threshold` = 25
import matplotlib.pyplot as plt
plt.axhline(y         = 25, 
            color     = 'black', 
            linewidth = 1)
plt.title("Single Linkage")


'''
Notice on the left, 
North Carolina, Florida, and Alaska have formed 3 clusters.
In other words, we have 3 clusters with ONLY ONE State!

And then we have orange cluster, green cluster, and red cluster!


But 3 clusters with only 1 State does NOT make much sense.
Let's try Ward Linkage!
'''













# =============================================================================
# Ward Linkage Dendrogram
# =============================================================================
'''
Process is very similar to Single Linkage Dendrogram

Just change linkage method to 'ward', and then
Determine a reasonable color_threshold (I chose 250 below)
'''

# ward linkage
ward_link = linkage(
    y      = usarrests.drop(['State'], axis = 1),
    method = 'ward')

# generate dendrogram using the single linkage above
dendrogram(Z               = ward_link,
           labels          = usarrests.State.tolist(),
           color_threshold = 250)

# good to draw horizontal line corresponding to `color_threshold` = 250
plt.axhline(y         = 250, 
            color     = 'black', 
            linewidth = 1)
plt.title('Ward Linkage')

# Now we have 3 clusters with roughly equal number of States!
# NO clusters with only 1 State















# =============================================================================
# Cluster Labeling using Ward Linkage Result
# =============================================================================

# import fcluster
from scipy.cluster.hierarchy import fcluster

# create cluster label with fcluster(), using ward linkage result
cluster_label = fcluster(Z         = ward_link, 
                         t         = 250,  # <- color_threshold role
                         criterion = 'distance')

# check our (Ward Linkage) cluster label
cluster_label # we have 3 labels for 3 clusters: 1, 2, 3

# add cluster labels to our usarrests
usarrests["cluster"] = cluster_label 
usarrests.head() # notice cluster labels are added


# Find mean Murder,Assault,Rape score for each cluster
usarrests.groupby('cluster').mean()[['Murder', 'Assault', 'Rape']]

'''
Cluster 1's Murder Assault Rape values are higher than 
those of cluster 2 and 3. 

So the States belonging to cluster 1 have relatively high crime rates.

On the other hand, the States in cluster 3 have relatively low crime rates.
'''





















"""
This is the end of "Dendrogram" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""



















