#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 14:03:34 2022

@author: rj
"""

# data download
# http://www.crbooks.co.kr/04_board/?mode=1&mcode=0404010000&page=1&no=&hd=&kind=all&keyword=%EA%B0%9C%EC%9D%B8%ED%99%94




import numpy as np

# set working directory
import os
os.chdir('/Users/rj/Desktop/RJstudio/PyRecommenderSystem')









# =============================================================================
# Data ch2.1
# =============================================================================


# u.users data: users
import pandas as pd
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('./book_data/u.user', 
                    sep      = '|', 
                    names    = u_cols, 
                    encoding ='latin-1')
users = users.set_index('user_id') # making user_id index
users.head()
users.shape   # (943, 4)
users.columns # ['age', 'sex', 'occupation', 'zip_code']



# u.item data: movies
i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL', 
          'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 
          'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('./book_data/u.item', 
                     sep      = '|', 
                     names    = i_cols, 
                     encoding ='latin-1')
movies = movies.set_index('movie_id') # making movie_id index
movies.head()
movies.shape  # (1682, 23)
movies.columns



# u.data data: ratings
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('./book_data/u.data', 
                      sep      = '\t', 
                      names    = r_cols, 
                      encoding = 'latin-1') 
ratings = ratings.set_index('user_id') # making user_id index
ratings.head()  # rating varies from 1 to 5
ratings.shape   # (100000, 3)     100K movie ratings
ratings.columns # ['movie_id', 'rating', 'timestamp']








# =============================================================================
# Best-seller Recommender ch2.2
# =============================================================================


# groupby movie_id of rating
ratings.groupby('movie_id').rating.mean()
movie_mean = ratings.groupby('movie_id').rating.mean() 
movie_mean

# sort by highest rating at the top
movie_mean.sort_values(ascending = False)
movie_mean.sort_values(ascending = False)[:5] # top 5 result below
'''
814     5.0
1599    5.0
1201    5.0
1122    5.0
1653    5.0
'''
foo = movie_mean.sort_values(ascending = False)[:5]
movies.loc[foo.index]['title']


def recom_movie1(n_items):
    movie_sort      = movie_mean.sort_values(ascending = False)[:n_items]
    recom_movies    = movies.loc[movie_sort.index]
    recommendations = recom_movies['title'] # just extract title
    return recommendations

recom_movie1(5)








# =============================================================================
# RMSE ch2.3
# =============================================================================


# RMSE
import numpy as np
def RMSE(y_true, y_pred):
    sq_diff = (np.array(y_true) - np.array(y_pred))**2
    sqrt_mean = np.sqrt(np.mean(sq_diff))
    return sqrt_mean

# quick test of RMSE()
RMSE([1,2,3], [1,1,1]) 
(np.array([1,2,3])- np.array([1,1,1]))**2 # array([0, 1, 4])
np.sqrt((0 + 1 + 4)/3)



set(ratings.index) # understanding set()
ratings.loc[34]
ratings.loc[34]['rating']
ratings.loc[34]['movie_id']
movie_mean[ratings.loc[34]['movie_id']]

rmse =[]
for user in set(ratings.index):
    y_true = ratings.loc[user]['rating']
    y_pred = movie_mean[ratings.loc[user]['movie_id']]
    accuracy = RMSE(y_true, y_pred)
    rmse.append(accuracy)
rmse
np.mean(rmse) # 0.996007224010567








# =============================================================================
# Groupby Recommendation ch2.4
# =============================================================================


# RELOAD data

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('./book_data/u.user', 
                    sep='|', 
                    names=u_cols, 
                    encoding='latin-1')
users.shape # (943, 5)


i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL', 'unknown', 
          'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 
          'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 
          'Thriller', 'War', 'Western']
movies = pd.read_csv('./book_data/u.item', 
                     sep='|', 
                     names=i_cols, 
                     encoding='latin-1')
movies.shape # (1682, 24)


r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('./book_data/u.data', 
                      sep='\t', 
                      names=r_cols, 
                      encoding='latin-1')
ratings.shape # (100000, 4)


# ----- RELOADing data done above


# selecting columns we need

# discard timestamp column
ratings = ratings.drop('timestamp',
                       axis = 1)
ratings.shape # (100000, 3)

# just extract 2 columns from movies
movies = movies[['movie_id', 'title']]
movies.shape # (1682, 2)



# train & test set 
from sklearn.model_selection import train_test_split

x = ratings.copy()
x   # a copy of ratings

y = ratings['user_id']
y   # just the user_id col

# stratified sampling 
x_train,x_test,y_train,y_test = train_test_split(x, 
                                                 y,
                                                 test_size = 0.25,
                                                 stratify  = y)

# RMSE function
def RMSE(y_true, y_pred):
    sq_diff = (np.array(y_true) - np.array(y_pred))**2
    sqrt_mean = np.sqrt(np.mean(sq_diff))
    return sqrt_mean


# to understand how zip() works
for (i, j) in zip(x_test['user_id'],x_test['movie_id']):
    print((i,j))
    
# function for scoring RMSE per model 🔥🔥
def score(model):
    id_pairs = zip(x_test['user_id'],x_test['movie_id']) # from test set
    y_pred   = np.array([model(user,movie) for (user,movie) in id_pairs])
    y_true   = np.array(x_test['rating']) # from test set
    return RMSE(y_true, y_pred)


# making a full matrix using x_train, using pivot()
rating_matrix = x_train.pivot(index   = 'user_id',
                              columns = 'movie_id',
                              values  = 'rating')
rating_matrix
rating_matrix.shape # (943, 1647) ; 943 users & 1647 movies


# movie_id grouped mean rating 
train_mean = x_train.groupby('movie_id')['rating'].mean()
train_mean
train_mean[2]



def best_seller(user_id, movie_id):
    """
    in score() function above, model needs 2 arguments
    """
    try:
        rating = train_mean[movie_id]
    except:
        """
        in case there are user_id & movie_id present in train data
        but not in test data, or vice versa
        """
        rating = 3.0
    return rating

score(best_seller) # 1.0227206332606449




# merging, using pd.merge()
x_train
users
merged_ratings = pd.merge(x_train, users)
merged_ratings # merged using commer user_id column

users = users.set_index('user_id')
users


# movie_id & sex grouped rating
g_mean = merged_ratings[['movie_id','sex','rating']].groupby(['movie_id', 'sex']).rating.mean()
g_mean
g_mean[1]
g_mean[1]['F']
'F' in g_mean[1] # True

1681 in rating_matrix # True
943 in rating_matrix  # True



def cf_gender(user_id, movie_id):
    if movie_id in rating_matrix:
        
        gender = users.loc[user_id]['sex']
        
        if gender in g_mean[movie_id]:
            gender_rating = g_mean[movie_id][gender]
        
        else: 
            gender_rating = 3.0
    
    else: 
        gender_rating = 3.0
    
    return gender_rating

score(cf_gender) # 1.032500733834306










