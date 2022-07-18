#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 15:36:03 2022

@author: rj
"""

import pandas as pd
import numpy as np


# =============================================================================
# RELOAD data
# =============================================================================


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



# =============================================================================
# selecting columns we need
# =============================================================================

# discard timestamp column
ratings = ratings.drop('timestamp',
                       axis = 1)
ratings.shape # (100000, 3)

# just extract 2 columns from movies
movies = movies[['movie_id', 'title']]
movies.shape # (1682, 2)




# =============================================================================
# train & test set 
# =============================================================================

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



# =============================================================================
# RMSE & score functions
# =============================================================================

# RMSE function
def RMSE(y_true, y_pred):
    sq_diff = (np.array(y_true) - np.array(y_pred))**2
    sqrt_mean = np.sqrt(np.mean(sq_diff))
    return sqrt_mean

    
# function for scoring RMSE per model 🔥🔥
def score(model):
    id_pairs = zip(x_test['user_id'],x_test['movie_id']) # from test set
    y_pred   = np.array([model(user,movie) for (user,movie) in id_pairs])
    y_true   = np.array(x_test['rating']) # from test set
    return RMSE(y_true, y_pred)


# =============================================================================
# full matrix
# =============================================================================

# making a full matrix using x_train, using pivot()
rating_matrix = x_train.pivot(index   = 'user_id',
                              columns = 'movie_id',
                              values  = 'rating')
rating_matrix
rating_matrix.shape # (943, 1647) ; 943 users & 1647 movies




##### NEW CODE starts from below



# =============================================================================
# cosine_similarity with everyone ch3.1
# =============================================================================

from sklearn.metrics.pairwise import cosine_similarity

rating_matrix # lots of NaN initially

# make a copy and replace NaN with 0
matrix_dummy = rating_matrix.copy().fillna(0)
matrix_dummy # to be used for cosine similarity

# cosine similarity with everyone 🔥🔥
user_similarity = cosine_similarity(X = matrix_dummy,
                                    Y = matrix_dummy)
user_similarity


# make user_similarity to a pandas dataframe
user_similarity = pd.DataFrame(user_similarity,
                               index   = rating_matrix.index,
                               columns = rating_matrix.index)
user_similarity
rating_matrix.index    # user_id
rating_matrix.columns  # user_id




3 in rating_matrix # True
user_similarity
user_similarity[3]
user_similarity[2:3]
s = user_similarity[1500]


rating_matrix
r = rating_matrix[1500] # 1500 is movie_id
rating_matrix.columns
type(r)
nul = r[r.isnull()].index
rating_matrix[1500].isnull()
r.dropna()
s.drop(nul)
np.dot(r.dropna(), s.drop(nul))
s.drop(nul).sum()

def CF_simple(user_id, movie_id):
    if movie_id in rating_matrix:
        
        # similarity with all other users
        sim_scores = user_similarity[user_id].copy()
        
        # how everyone rated a particular movie
        movie_ratings = rating_matrix[movie_id].copy()
        
        # index of users who didn't rate for a particular movie
        none_rating_idx = movie_ratings[movie_ratings.isnull()].index
        
        # get rid of all NaNs
        movie_ratings = movie_ratings.dropna()
        
        # also drop all similarities where movie_rating is NaN
        sim_scores = sim_scores.drop(none_rating_idx)
        
        # mean rating check p.34 note
        mean_rating = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
    
    else: 
        mean_rating = 3.0
    
    return mean_rating

score(CF_simple)
        
    
# =============================================================================
# KNN CF ch3.2
# =============================================================================


# modify score fuction so that it can take neighbor_size
def score1(model, neighbor_size = 0):
    id_pairs = zip(x_test['user_id'],x_test['movie_id']) # from test set
    y_pred   = np.array([model(user,movie,neighbor_size) for (user,movie) in id_pairs])
    y_true   = np.array(x_test['rating']) # from test set
    return RMSE(y_true, y_pred)



rating_matrix.shape # (943, 1643)
np.argsort(np.array(user_similarity[800].dropna()))
foo = pd.Series([3,1,10,5])
foo
foo_np = np.array(foo)
foo_np                               # [ 3,  1, 10,  5]
foo_np_idx = np.argsort(foo_np)      # [1, 0, 3, 2]
foo_np[foo_np_idx]      # [ 1,  3,  5, 10]
foo_np[foo_np_idx][-2:] # [ 5, 10]

def cf_knn(user_id, movie_id, neighbor_size = 0):
    if movie_id in rating_matrix:
        sim_scores = user_similarity[user_id].copy()
        movie_ratings = rating_matrix[movie_id].copy()
        none_rating_idx = movie_ratings[movie_ratings.isnull()].index
        movie_ratings = movie_ratings.drop(none_rating_idx)
        sim_scores = sim_scores.drop(none_rating_idx)
        
        # up to here pretty much same as CF_simple()
        
        
        # if Neighbor Size is not given
        # do the same calculation as CF_simple()
        if neighbor_size == 0:
            mean_rating = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
        
        
        # if Neighbor Size IS given neighbor_size != 0
        else:
            if len(sim_scores) > 1:
                neighbor_size = min(neighbor_size, len(sim_scores))
                
                # need to convert to np array to use np.argsort later
                sim_scores = np.array(sim_scores)
                movie_ratings = np.array(movie_ratings)
                # np. argsort
                user_idx = np.argsort(sim_scores)
                
                # get top sim_scores, movie_ratings
                sim_scores = sim_scores[user_idx][-neighbor_size:]
                movie_ratings = movie_ratings[user_idx][-neighbor_size:]
                
                mean_rating = np.dot(sim_scores, movie_ratings)/sim_scores.sum()
            
            else:
                mean_rating = 3.0
    else: 
        mean_rating = 3.0
    
    return mean_rating

# test
cf_knn(1,2,10)

# use score1
score1(cf_knn, neighbor_size=30) # 1.0118687253305454




# let's use ENTIRE data this time
rating_matrix = ratings.pivot_table(values = 'rating',
                                    index  = 'user_id',
                                    coloumns = 'moive_id')
rating_matrix.shape # (943, 1643)

matrix_dummy = rating_matrix.copy().fillna(0)
matrix_dummy
user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
user_similarity = pd.DataFrame(user_similarity,
                               index   = rating_matrix.index,
                               columns = rating_matrix.index)

user_similarity
rating_matrix
type(rating_matrix) # pandas.core.frame.DataFrame
rating_matrix.loc[1]



foo = rating_matrix.loc[1]
foo
foo.loc[1682] # nan

def recom_movie(user_id, n_items, neighbor_size = 30):
    
    # just one user's movie rating
    user_movie = rating_matrix.loc[user_id].copy()
    
    for movie in rating_matrix:
        if pd.notnull(user_movie.loc[movie]):
            user_movie.loc[movie] = 0
        else: 
            # using cf_knn() from above
            user_movie.loc[movie] = cf_knn(user_id, movie, neighbor_size)
        
    movie_sort = user_movie.sort_values(ascending = False)[:n_items]
    recom_movies = movies.loc[movie_sort.index]
    recommendations = recom_movies['title']
    return recommendations


recom_movie(user_id = 2, n_items = 5, neighbor_size= 30)



# ------

# now we need to decide optimal neighbor size

# https://thepythonguru.com/python-string-formatting/
# interesting print format in python
print('integer: %d float: %.4f' % (3, 2.1))



rating_matrix = x_train.pivot_table(
    values  = 'rating',
    index   = 'user_id',
    columns = 'movie_id')
rating_matrix

matrix_dummy = rating_matrix.copy().fillna(0)
user_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
user_similarity.shape # (943, 943)
user_similarity = pd.DataFrame(data = user_similarity,
                               index = rating_matrix.index,
                               columns = rating_matrix.index)

for neighbor_size in [10, 20, 30, 40, 50, 60]:
    print("Neighbor Size = %d / RMSE = %.4f" % (neighbor_size, score1(cf_knn, neighbor_size)))

'''
Neighbor Size = 10 / RMSE = 1.0250
Neighbor Size = 20 / RMSE = 1.0087
Neighbor Size = 30 / RMSE = 1.0064
Neighbor Size = 40 / RMSE = 1.0056
Neighbor Size = 50 / RMSE = 1.0058
Neighbor Size = 60 / RMSE = 1.0059
'''



# =============================================================================
# user bias ch3.3
# =============================================================================

foo = np.array([[1,2,3],
               [1,2,3]])
boo = np.array([1,1,1])

foo - boo

foo.shape # (2, 3)
boo.shape # (3,)

# each user's mean rating
rating_mean = rating_matrix.mean(axis = 1)
rating_mean[1]
rating_mean.shape # (943,)

rating_matrix.shape # (943, 1649)
rating_matrix.T.shape # (1649, 943)
rating_bias = (rating_matrix.T - rating_mean).T
rating_bias


def CF_knn_bias(user_id, movie_id, neighbor_size = 0):
    if movie_id in rating_bias:
        sim_scores = user_similarity[user_id].copy()
        movie_ratings = rating_bias[movie_id].copy()
        none_rating_idx = movie_ratings[movie_ratings.isnull()].index
        movie_ratings = movie_ratings.drop(none_rating_idx)
        sim_scores = sim_scores.drop(none_rating_idx)
        
        if neighbor_size == 0:
            prediction = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
            prediction = prediction + rating_mean[user_id]
        
        else: 
            if len(sim_scores) > 1 :
                neighbor_size = min(neighbor_size, len(sim_scores))
                sim_scores = np.array(sim_scores)
                movie_ratings = np.array(movie_ratings)
                user_idx = np.argsort(sim_scores)
                sim_scores = sim_scores[user_idx][-neighbor_size:]
                movie_ratings = movie_ratings[user_idx][-neighbor_size:]
                prediction = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
                prediction = prediction + rating_mean[user_id]
                
            
            else: 
                prediction = rating_mean[user_id]
    
    else:
        prediction = rating_mean[user_id]

    return prediction


score1(CF_knn_bias, 30) # 0.943742791711317



# =============================================================================
# Significance level ch3.4
# =============================================================================

rating_matrix
rating_matrix>0
(rating_matrix>0).astype(float)

rating_binary1 =  np.array((rating_matrix>0).astype(float))
rating_binary1.shape # (943, 1649)
rating_binary1

rating_binary2 = rating_binary1.T
rating_binary2.shape # (1649, 943)
rating_binary2

counts = np.dot(rating_binary1, rating_binary2)
counts.shape #  (943, 943)
counts

counts = pd.DataFrame(data = counts, 
             index = rating_matrix.index,
             columns = rating_matrix.index).fillna(0)
counts
counts[943]
counts[943] < 3

SIG_LEVEL = 3
MIN_RATINGS = 2

def CF_knn_bias_sig(user_id, movie_id, neighbor_size = 0):
    if movie_id in rating_bias:
        sim_scores = user_similarity[user_id].copy()
        movie_ratings = rating_bias[movie_id].copy()
        
        no_rating = movie_ratings.isnull()
        common_counts = counts[user_id]
        low_significance = common_counts < SIG_LEVEL
        none_rating_idx = movie_ratings[no_rating | low_significance].index
        
        movie_ratings = movie_ratings.drop(none_rating_idx)
        sim_scores = sim_scores.drop(none_rating_idx)
        
        if neighbor_size == 0:
            prediction = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
            prediction = prediction + rating_mean[user_id]
        
        else: 
            if len(sim_scores) > MIN_RATINGS :
                neighbor_size = min(neighbor_size, len(sim_scores))
                sim_scores = np.array(sim_scores)
                movie_ratings = np.array(movie_ratings)
                user_idx = np.argsort(sim_scores)
                sim_scores = sim_scores[user_idx][-neighbor_size:]
                movie_ratings = movie_ratings[user_idx][-neighbor_size:]
                prediction = np.dot(sim_scores, movie_ratings) / sim_scores.sum()
                prediction = prediction + rating_mean[user_id]
                
            else: 
                prediction = rating_mean[user_id]
    
    else:
        prediction = rating_mean[user_id]

    return prediction


score1(CF_knn_bias_sig, 30) # 0.856329119014677




# =============================================================================
# Item-Based CF ch3.5
# =============================================================================

rating_matrix

rating_matrix_t = np.transpose(rating_matrix)
rating_matrix_t.shape # (1649, 943)
rating_matrix_t[900]

matrix_dummy = rating_matrix_t.copy().fillna(0)

item_similarity = cosine_similarity(matrix_dummy, matrix_dummy)
item_similarity.shape # (1649, 1649)
item_similarity = pd.DataFrame(data = item_similarity,
                               index = rating_matrix_t.index,
                               columns = rating_matrix_t.index)
item_similarity



def CF_IBCF(user_id, movie_id):
    if movie_id in item_similarity:
        sim_scores = item_similarity[movie_id]
        user_rating = rating_matrix_t[user_id]
        non_rating_idx = user_rating[user_rating.isnull()].index
        user_rating = user_rating.dropna()
        sim_scores = sim_scores.drop(non_rating_idx)
        mean_rating = np.dot(sim_scores, user_rating) / sim_scores.sum()
    else:
        mean_rating = 3.0
    return mean_rating

score(CF_IBCF) # 0.9701331666890655















