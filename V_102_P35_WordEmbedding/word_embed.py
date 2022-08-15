#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 15:25:29 2022

@author: rj
"""


'''
Like all other neural networks, deep-learning models don’t take 
as input raw text:
they only work with numeric tensors. Vectorizing text is the process of 
transforming text into numeric tensors.


According to 
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
Word embeddings can be thought of as an alternate to one-hot encoding 
along with dimensionality reduction.




The Embedding layer takes at least two
arguments: the number of possible tokens
and the dimensionality of the embeddings


https://stackoverflow.com/questions/61848825/why-is-input-length-needed-in-layers-embedding-in-keras-tensorflow
In the Embedding layer we put input_length
By specifying the dimension, you're making sure the model receives 
fixed-length input.
 if you don't specify the input shape in the Input layer and also not in 
 the embedding layer, there's no way the model can be built with the proper 
 set of parameters.
'''


import numpy as np

from tensorflow.keras.backend import clear_session
clear_session()














"""
Word Embeddings
 (▰˘◡˘▰)


Word Embeddings is "a popular and powerful way 
to associate a vector with a word" 🚀


An excellent benefit to using Word Embeddings is that 
it can "pack more information into far FEWER dimensions" than
the "word vectors obtained via One-Hot encoding" 


I like the following remark about Word Embeddings 😊
from https://medium.com/aiguys/word-embeddings-cbow-and-skip-gram-5d615ad61d3d 
    "Word embeddings can be thought of as an alternate to one-hot encoding 
    along with dimensionality reduction."


A lot of explanation in this video is from a book titled,
"Deep Learning with Python" (Chapter 6)


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""




















# =============================================================================
# Load IMDB Movie Review data
# =============================================================================

# Load IMDB data 
from tensorflow.keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
# restricted movie reviews to top 10000 most common words



# Quick Data Exploration below:

# 9999 is the max word index due to `num_words=10000` above
max([max(sequence) for sequence in x_train]) # 9999


# train data
x_train.shape # (25000,) ; 25K movie reviews 
y_train.shape # (25000,) ; 25K positive or negative review classifications
np.unique(y_train, return_counts = True)
# Only 2 unique values: 0 & 1
# 12500 0's & 12500 1's
# 0 for negative & 1 for positive review


# test data
x_test.shape # (25000,)
y_test.shape # (25000,)
np.unique(y_test, return_counts = True)
# identical structure as the train data! 




















# =============================================================================
# Preprocessing (pad_sequences)
# =============================================================================

# import preprocessing
from tensorflow.keras import preprocessing

# cut off the reviews after only 20 words
# for train data
x_train_20 = preprocessing.sequence.pad_sequences(sequences = x_train, 
                                                  maxlen    = 20)
# for test data
x_test_20  = preprocessing.sequence.pad_sequences(sequences = x_test, 
                                                  maxlen    = 20)


# check with first review
len(x_train[0])    # 218, previously 218 words
len(x_train_20[0]) # 20,  NOW only 20 words, good ✅






# my own check
for i,j in enumerate(x_train): # works
    print(i, j)
    break

for i,j in x_train:    # not work
    print(i, j)
    break




len(x_train[24999])
x_train.shape      # (25000,)
x_train_20.shape   # (25000, 20)
len(x_train_20[24999])


















# =============================================================================
# Embedding layer 🔥 (inside network architecture)
# =============================================================================

# import what I need :) 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding


# Start building linear stack of layers, using Sequential()
wordEmbed = Sequential()

# Embedding layer 🔥
# Let network learn 8-dim embeddings for each of the 10K words
# to ensure model receives fixed-length input, I put `input_length = 20`
wordEmbed.add(Embedding(input_dim    = 10000,  # 10K words
                        output_dim   = 8,      # 8-dim embeddings
                        input_length = 20)) 
'''
From Textbook (left for viewers to read after pausing ⭐⭐):

The Embedding layer is best understood as a dictionary that maps 
integer indices (which stand for specific words) to dense vectors. 

It takes integers as input, looks up these integers in an internal dictionary, 
and it returns the associated vectors. 
It’s effectively a dictionary lookup
'''



# Flatten() the 3D tensor of embeddings into a 2D tensor
wordEmbed.add(Flatten())


# Dense layer with 1 unit & sigmoid activation
wordEmbed.add(Dense(units      = 1,
                    activation = 'sigmoid'))
'''
Essentially, we add binary classifier (sigmoid) to find if review was
0 (for negative) OR 1 (for positive)
'''


# check network architecture
wordEmbed.summary()
















# =============================================================================
# Compile & Fit
# =============================================================================

# Compilation; determine appropriate 1) optimizer 2) loss 3) metrics
wordEmbed.compile(
    optimizer = 'rmsprop', 
    loss      = 'binary_crossentropy',
    metrics   = ['acc']
    )

# Fit
wordEmbed_fit = wordEmbed.fit(
    x                = x_train_20, # training data
    y                = y_train,
    epochs           = 15, 
    batch_size       = 32,
    validation_split = 0.2
    )


# View Training Loss & Accuracy, using history
wordEmbed_fit.history

# Let me visualize Training result next!















# =============================================================================
# Plot Training Result
# =============================================================================

# need matplotlib
import matplotlib.pyplot as plt

# Plot
plt.plot([i+1 for i in range(15)],
         wordEmbed_fit.history['acc'],
         label = "Training Accuracy")
plt.plot([i+1 for i in range(15)],
         wordEmbed_fit.history['val_acc'],
         label = "Validation Accuracy")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")


'''
Training Accuracy increases monotonically as we expect
Validation Accuracy reaches around 75% max

It is not an "excellent" accuracy, but it is acceptable considering
how ONLY first 20 words for each review were used 
'''























"""
This is the end of "Word Embeddings" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""

















