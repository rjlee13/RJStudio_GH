#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 18:30:49 2022

@author: rj
"""

import os
os.chdir('/Users/rj/Desktop/RJstudio/PyRecommenderSystem')


from tensorflow.keras.backend import clear_session
clear_session()

# =============================================================================
# Load Data
# =============================================================================


import pandas as pd
import numpy as np

# csv 파일에서 불러오기
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('./book_data/u.data', 
                      names=r_cols,  sep='\t',encoding='latin-1')
ratings = ratings[['user_id', 'movie_id', 'rating']].astype(int)            # timestamp 제거

ratings.shape # (100000, 3)


# train test 분리
from sklearn.utils import shuffle
TRAIN_SIZE = 0.75
ratings = shuffle(ratings)
cutoff = int(TRAIN_SIZE * len(ratings))
cutoff # 75000

ratings_train = ratings.iloc[:cutoff]
ratings_test = ratings.iloc[cutoff:]



ratings_train.shape # (75000, 3)
ratings_test.shape  # (25000, 3)


ratings_train.columns # 'user_id', 'movie_id', 'rating'
ratings_train.user_id
ratings_train.user_id.values







# =============================================================================
# MF in Keras ch6.1
# =============================================================================

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten # 🔥
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adamax

# Variable Initialization  
K = 200                             # Number of Latent factor
mu = ratings_train.rating.mean()    # entire average
M = ratings.user_id.max() + 1       # Number of users
N = ratings.movie_id.max() + 1      # Number of movies

mu, M, N # (3.527826666666667, 944, 1683)

def RMSE(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))



user = Input(shape=(1,))
item = Input(shape=(1,))
user, item


P_embedding = Embedding(M,K, embeddings_regularizer=l2())(user)
Q_embedding = Embedding(N,K, embeddings_regularizer=l2())(item)
'''
Word embeddings can be thought of as an alternate to one-hot encoding along with dimensionality reduction.
'''
user_bias = Embedding(M,1, embeddings_regularizer=l2())(user)
item_bias = Embedding(N,1, embeddings_regularizer=l2())(item)


R = layers.dot([P_embedding, Q_embedding], axes = 2)
R = layers.add([R, user_bias, item_bias])
R = Flatten()(R)


'''
check
https://www.tensorflow.org/guide/keras/functional 
for understanding code!
'''
model = Model(inputs = [user, item], outputs = R)
model.compile(loss = RMSE,
              optimizer = SGD(),
              metrics = [RMSE])

model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)


# =============================================================================
# Model Fitting
# =============================================================================


result = model.fit(
    x= [ratings_train.user_id.values, ratings_train.movie_id.values],
    y= ratings_train.rating.values - mu,
    epochs = 30,
    batch_size = 256,
    validation_data = (
        [ratings_test.user_id.values, ratings_test.movie_id.values],
        ratings_test.rating.values-mu)
    )

# =============================================================================
# Plot RMSE
# =============================================================================

import matplotlib.pyplot as plt
plt.plot(result.history['RMSE'], label = "Train RMSE")
plt.plot(result.history['val_RMSE'], label = "Test RMSE")
plt.legend()


user_ids = ratings_test.user_id.values[0:6]
movie_ids = ratings_test.movie_id.values[0:6]

model.predict([user_ids,movie_ids]) + mu








# =============================================================================
# More Hidden layers ch6.2
# =============================================================================


# Keras model
user = Input(shape=(1, ))                                               # User input
item = Input(shape=(1, ))                                               # Item input
P_embedding = Embedding(M, K, embeddings_regularizer=l2())(user)        # (M, 1, K)
Q_embedding = Embedding(N, K, embeddings_regularizer=l2())(item)        # (N, 1, K)
user_bias = Embedding(M, 1, embeddings_regularizer=l2())(user)          # User bias term (M, 1, )
item_bias = Embedding(N, 1, embeddings_regularizer=l2())(item)          # Item bias term (N, 1, )


# concatenate layers
from tensorflow.keras.layers import Dense, Concatenate, Activation
P_embedding = Flatten()(P_embedding)
Q_embedding = Flatten()(Q_embedding)
user_bias = Flatten()(user_bias)
item_bias = Flatten()(item_bias)
R = Concatenate()([P_embedding, Q_embedding, user_bias, item_bias])


# Neural Network
R = Dense(units = 2048)(R)
R = Activation('linear')(R)
R = Dense(units = 256)(R)
R = Activation('linear')(R)
R = Dense(units = 1)(R)

model = Model(inputs = [user,item], 
              outputs = R)


model.compile(
    loss = RMSE,
    optimizer = SGD(),
    metrics = [RMSE]
    )

model.summary()

result = model.fit(
    x = [ratings_train.user_id.values, ratings_train.movie_id.values],
    y = ratings_train.rating.values - mu,
    epochs = 50,
    batch_size = 512,
    validation_data = ([ratings_test.user_id.values, ratings_test.movie_id.values],
                       ratings_test.rating.values - mu))


# plot
import matplotlib.pyplot as plt
plt.plot(result.history['RMSE'], label = "Train RMSE")
plt.plot(result.history['val_RMSE'], label = "Test RMSE")
plt.legend()





# =============================================================================
# Occupation ch6.3
# =============================================================================


u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('./book_data/u.user', sep='|', 
                    names=u_cols, encoding='latin-1')
users = users[['user_id', 'occupation']]

users.shape # (943, 2)

# convert occupation from string to integer
occupation = {}
def convert_occ(x):
    if x in occupation:
        return occupation[x]
    else:
        occupation[x] = len(occupation)
        return occupation[x]

foo = {'a': [1,2,3],
 'b': [1,1,1]}

foo['a']
foo['c'] = 23
foo


users['occ'] = users['occupation'].apply(convert_occ)
users


L= len(occupation)
L # 21


train_occ = pd.merge(ratings_train, users, on = 'user_id')['occ']
train_occ
test_occ = pd.merge(ratings_test, users, on = 'user_id')['occ']
test_occ




# Keras model
user = Input(shape=(1, ))
item = Input(shape=(1, ))
P_embedding = Embedding(M, K, embeddings_regularizer=l2())(user)
Q_embedding = Embedding(N, K, embeddings_regularizer=l2())(item)
user_bias = Embedding(M, 1, embeddings_regularizer=l2())(user)
item_bias = Embedding(N, 1, embeddings_regularizer=l2())(item)

# Concatenate layers
from tensorflow.keras.layers import Dense, Concatenate, Activation
P_embedding = Flatten()(P_embedding)
Q_embedding = Flatten()(Q_embedding)
user_bias = Flatten()(user_bias)
item_bias = Flatten()(item_bias)

occ = Input(shape = (1,))
occ_embedding = Embedding(L, 3, embeddings_regularizer=l2())(occ)
occ_layer = Flatten()(occ_embedding)

R = Concatenate()([P_embedding,Q_embedding, user_bias, item_bias, occ_layer])

# Neural network
R = Dense(2048)(R)
R = Activation('relu')(R)
R = Dense(256)(R)
R = Activation('linear')(R)
R = Dense(1)(R)

model = Model(inputs = [user,item,occ], outputs = R)


model.compile(
  loss=RMSE,
  optimizer=SGD(),
  #optimizer=Adamax(),
  metrics=[RMSE]
)
model.summary()

# Model fitting
result = model.fit(
  x=[ratings_train.user_id.values, ratings_train.movie_id.values, train_occ.values],
  y=ratings_train.rating.values - mu,
  epochs=50,
  batch_size=512,
  validation_data=(
    [ratings_test.user_id.values, ratings_test.movie_id.values, test_occ.values],
    ratings_test.rating.values - mu
  )
)


import matplotlib.pyplot as plt
plt.plot(result.history['RMSE'], label="Train RMSE")
plt.plot(result.history['val_RMSE'], label="Test RMSE")
plt.xlabel('epoch')
plt.ylabel('RMSE')
plt.legend()







