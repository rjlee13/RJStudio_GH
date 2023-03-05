#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 23:18:27 2023

@author: rj
"""












"""
Decision Tree Classifier 🌲
 (▰˘◡˘▰)

In this video, I train a Decision Tree Classifier using Scikit-learn 🚀

Decision Trees can be used for BOTH regression and classification problems.
For this video, I use Decision Tree to 'classify' 3 wine classes. 


Let me show you a saved image of Decision Tree I trained 
with this video's code!


Decision Trees are very easy to interpret and can be displayed graphically.👍
However, they do NOT have the same predictive accuracy as other Machine 
Learning algorithms. 😔



Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""



















# =============================================================================
# Wine Dataset for Classification
# =============================================================================

# import Wine Dataset
from sklearn.datasets import load_wine

# load Wine Dataset 
wine = load_wine()

# see our (X-variable) feature names & data
wine.feature_names 
wine.data

# create a Pandas DataFrame using wine.feature_names & data
import pandas as pd # import pandas
wine_pd = pd.DataFrame(data    = wine.data,
                       columns = wine.feature_names)
wine_pd # check our Pandas DataFrame

# see our target (Y-variable)
wine.target_names # there are 3 CLASSES, to be predicted by Decision Tree soon 
wine.target       # 3 classes: 0, 1, 2

# create a new column called 'class' for wine.target
wine_pd['class'] = wine.target
wine_pd['class'] = wine_pd['class'].map({0:'class_0', 1:'class_1', 2:'class_2'})

wine_pd # notice there is a NEW column called class

# see distribution of class
wine_pd.groupby('class').size()
# there are 71 class_1 (most)  ;  there are 48 class_2 (least) 





















# =============================================================================
# Generate Train & Test data
# =============================================================================

# X-variables are all variables except 'class'
x_var = list(wine_pd.columns.difference(['class']))
x_var # list of all variables except 'class'
X = wine_pd[x_var]
X     # NO 'class' column

# Y-variable is 'class', to be predicted later by Decision Tree
Y =  wine_pd['class']
Y # class data


# import train_test_split
from sklearn.model_selection import train_test_split

# Generate Train & Test data, using train_test_split()
x_train, x_test, y_train, y_test = train_test_split(
    X, Y,
    test_size    = 0.3,
    stratify     = Y,   # note stratified split is used
    random_state = 3)


# check Train data
x_train # all features' data except 'class'
y_train # class
y_train.groupby(y_train).size() #(distribution) class_1 most ; class_2 least


# check Test data
x_test # all features' data except 'class'
y_test # class
y_test.groupby(y_test).size()   #(distribution) class_1 most ; class_2 least




















# =============================================================================
# Decision Tree 🌲
# =============================================================================

# import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# create an instance of DecisionTreeClassifier, dtree 🌲
dtree = DecisionTreeClassifier(
    max_depth         = 5, # max depth of tree
    min_samples_split = 3  # min sample numbers to split internal node
    )
# fit / train our Decision Tree model (dtree) using Train data!
dtree.fit(x_train, y_train)


# predict classes with our dtree using Test data, x_test
dtree_pred = dtree.predict(x_test)
dtree_pred # see our prediction


# Evaluate how well our Decision Tree predicted!
# import confusion_matrix & accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score

# let's evaluate using Confusion Matrix
cm = confusion_matrix(y_test, dtree_pred)
cm # our prediction (dtree_pred) and true y_test agree  15+17+15 times
   # they DISAGREE 3+4 times

# check accuracy 
acc = accuracy_score(y_test, dtree_pred)
acc 

















# =============================================================================
# Feature Importance
# =============================================================================
'''
Feature Importance assigns a score to each feature/variable.
A feature's high score means it had a large effect on the model trained,
and vice versa.
'''

# check feature importance of our Decision Tree, dtree
fi = dtree.feature_importances_
fi_pd = pd.DataFrame(data = fi) # turn it into pandas DataFrame
fi_pd # hmm... would be nice to have feature names next to their scores!  

# create pandas DataFrame of feature names
column_pd = pd.DataFrame(data = X.columns)
column_pd # feature names!


# now concatenate (pd.concat) feature importance & feature names
feature_importance = pd.concat([column_pd, fi_pd],
                               axis = 1)
feature_importance.columns = ['feature_name', 'importance']

# Sort by feature importance 
feature_importance.sort_values(by        = ['importance'], 
                               ascending = False)

# proline & color_intensity had the largest effect 
























# =============================================================================
# Visualize Decision Tree! 
# =============================================================================

# import what I need
import numpy as np
import pydotplus
from sklearn.tree import export_graphviz

# use export_graphviz() function
dot_data = export_graphviz(
    decision_tree = dtree, # our Decision Tree model
    feature_names = x_var, # feature names
    class_names   = np.array(['class_0', 'class_1', 'class_2']), # 3 classes
    filled        = True,
    rounded       = True)

# now visualize
dot_graph = pydotplus.graph_from_dot_data(dot_data)
from IPython.display import Image
Image(dot_graph.create_png())

# hard to see all in the console...
# so I saved it as PNG file!


# notice that 'proline' and 'color_intensity' are at the top of the tree!
# they are the 2 features that received the highest Feature Importance!!



















"""
This is the end of "Decision Tree Classifier" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""






















