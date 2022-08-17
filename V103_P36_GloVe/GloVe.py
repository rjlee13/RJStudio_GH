#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 11:12:53 2022

@author: rj
"""

import numpy as np


from tensorflow.keras.backend import clear_session
clear_session()













"""
GloVe, Global Vectors for Word Representation 
 (▰˘◡˘▰)



When your training data is too small,
"you canNOT use your data alone to learn an appropriate task-specific 
embedding of your vocabulary."

In such cases,
"it makes sense to REUSE features learned on a different problem." 🚀


There are various "PRETRAINED Word Embeddings", and
GloVe🔥 is a very popular one. 
It was developed by Stanford in 2014. 


In this video, I show how to load GloVe Embeddings to your network~! 🔥
(I am NOT aiming for high accuracy!)


A lot of explanation in this video is from a book titled,
"Deep Learning with Python" (Chapter 6)


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""

















# =============================================================================
# Load & Preprocess IMDB data
# =============================================================================

# Load IMDB data 
# Consider only the top 10,000 common words in the dataset
from tensorflow.keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 10000)



# check train data
x_train.shape # (25000,) ; 25K movie reviews 
y_train.shape # (25000,) ; 25K positive or negative review classifications
np.unique(y_train, return_counts = True)
# Only 2 unique labels in y_train: 0 & 1
# 12500 0's & 12500 1's
# 0 for negative & 1 for positive review
# So, this is a BINARY classification problem


# Preprocess
from tensorflow.keras import preprocessing

# Cut off reviews after 100 words by setting `maxlen = 100`
# train data
x_train_100 = preprocessing.sequence.pad_sequences(sequences = x_train, 
                                                   maxlen    = 100)
# test data
x_test_100  = preprocessing.sequence.pad_sequences(sequences = x_test, 
                                                   maxlen    = 100)


















# =============================================================================
# Download GloVe!
# =============================================================================

'''
Go to: https://nlp.stanford.edu/projects/glove

then click on: glove.6B.zip

Warning!🚨 Zip file is 800MB+

Unzip the downloaded file 
Warning!🚨 once unzipped, it is over 2GB
'''

























# =============================================================================
# Parsing the GloVe 🪄 Embedding Index
# =============================================================================

# empty dictionary to start with
embedding_index = {}

# Path / Directory to my GloVe
glove_dir = '/Users/rj/Desktop/RJstudio/V103/glove.6B/glove.6B.100d.txt'

# populate embedding_index dictionary
    # 1) words   will be dict (embedding_index) keys
    # 2) vectors will be dict (embedding_index) values
f = open(glove_dir) # open glove
for line in f:
   values = line.split() 
   word = values[0]                                 # word   as keys
   coefs = np.asarray(values[1:], dtype='float32')  # vector as values
   embedding_index[word] = coefs                    # populate embedding_index
f.close()


# check key & value of a third-index word from embedding_index created
list(embedding_index.keys())[3]    # third word  is "of"
list(embedding_index.values())[3]  # word vector for "of"


# get corresponding vector for another word, "good"
embedding_index.get("good")

# It is 100-dimensional; that's why there is '100d' in glove.6B.100d.txt
len(embedding_index.get("good")) # 100

# total number of word vectors in embedding_index
print('There are %s word vectors.' % len(embedding_index)) # 400K words




















# =============================================================================
# IMDB Word Index 📖
# =============================================================================

# obtain IMDB dataset Word Index, using get_word_index()
word_index = imdb.get_word_index()
# check word index
word_index  
'''
Basically, it is a dictionary that MAPS words to an integer index

'tingled' has integer index 64027
'''





























# =============================================================================
# Embedding Matrix 🔮
# =============================================================================
'''
Using `embedding_index` & `word_index` created from above, 
we build Embedding Matrix that will be loaded to Embedding Layer later!
'''

# Keep using 100-dimensional
embedding_dim = 100
# start with ALL zero Embedding Matrix
embedding_matrix = np.zeros((10000, embedding_dim)) 
# 0's will be REPLACED by appropriate word vectors in FOR loop below


# FOR loop: fill embedding_matrix with appropriate word vectors
for word, i in word_index.items():
    if i < 10000:  # consider only the top 10,000 common words
        embedding_vector = embedding_index.get(word) # vector for word
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector   # REPLACE 0's with vectors



# (Quick) checking Embedding Matrix:
    
word_index.get('fantastic')      # 'fantastic' is 774th most frequent word
embedding_index.get('fantastic') # GloVe word vector for 'fantastic'
# notice LAST 2 numbers are: 2.5169e-02,  5.5207e-01

# our Embedding Matrix should have the SAME word vector at 774th row
embedding_matrix[774] # 2.51690000e-02,  5.52070022e-01
# get the same LAST 2 numbers from our embedding_matrix too! good ✅















# =============================================================================
# build Architecture
# =============================================================================

# import what I need
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense


# start building linear stack of layers, using Sequential()
gloveEmbed = Sequential()

# Embedding layer  (GloVe Embedding Matrix NOT loaded yet)
gloveEmbed.add(Embedding(input_dim    = 10000, # 10K words
                         output_dim   = 100,   # 100-dim embeddings
                         input_length = 100))
'''
Note I made a video explaining Embedding Layer before.
If you want to see more detail on Word Embedding, please check the video :) 
(YouTube link in Description)
'''


# Flatten() the 3D tensor of embeddings into a 2D tensor
gloveEmbed.add(Flatten())

# Dense layer, acting as a binary classifier (sigmoid activation)
gloveEmbed.add(Dense(units      = 1,
                     activation = 'sigmoid'))

# check model
gloveEmbed.summary()

# so far nothing new... 
# NEXT let's load Embedding Matrix to Embedding Layer!🔥













# =============================================================================
# Load GloVe Embeddings 🔥🔥
# =============================================================================

# View 3 layers we specified above
gloveEmbed.layers
# 1) Embedding 2) Flatten 3) Dense


# we can choose Embedding layer like this
gloveEmbed.layers[0] # keras.layers.embeddings.Embedding


# Load GloVe word embeddings (embedding_matrix) into the Embedding layer 🔥🔥
# using set_weights()
gloveEmbed.layers[0].set_weights([embedding_matrix])


# "Freeze" 🧊 the Embedding layer
gloveEmbed.layers[0].trainable = False

'''
Pretrained parts should NOT be updated during training 
to avoid forgetting what they already know!! That is why we Freeze!
'''

















# =============================================================================
# Compile & Fit
# =============================================================================

# Compilation; determine appropriate 1) optimizer 2) loss 3) metrics
gloveEmbed.compile(
    optimizer = 'rmsprop',
    loss      = 'binary_crossentropy',
    metrics   = ['acc']
    )


# Fit
glove_Embed_fit = gloveEmbed.fit(
    x                = x_train_100, # train data
    y                = y_train,     # remember reviews are cut after 100 words
    epochs           = 15,
    batch_size       = 32,
    validation_split = 0.2
    )

# Let me plot training result















# =============================================================================
# Plot Training / Validation Accuracies
# =============================================================================

# import matplotlib
import matplotlib.pyplot as plt

# Plot
plt.plot([i+1 for i in range(15)],
         glove_Embed_fit.history['acc'],
         label = "Training Accuracy")
plt.plot([i+1 for i in range(15)],
         glove_Embed_fit.history['val_acc'],
         label = "Validation Accuracy")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

'''
Validation Accuracy stays a little below 60%, which is not very accurate

That is NOT surprising because:

    1) Remember there are only 3 layers (Embedding, Flatten, Dense),
    so there are NOT many layers that can learn!

    2) I also cut off reviews after 100 words.

    3) On top of that, I FROZE 🧊 the Embedding (pretrained GloVe) layer! 
'''














# =============================================================================
# What happens when Embedding layer is NOT frozen 🧊?
# =============================================================================
'''
We 'should' be freezing  🧊 pretrained layers, like GloVe Embedding weights

But...
I was curious to see the change when we do NOT freeze the Embedding layer. 
I am giving more learning capacity this way!
Just an experiment 🫠
'''
noFreeze = Sequential()
noFreeze.add(Embedding(input_dim    = 10000, 
                       output_dim   = 100, 
                       input_length = 100))
noFreeze.add(Flatten())
noFreeze.add(Dense(units      = 1,
                   activation = 'sigmoid'))



noFreeze.layers[0].set_weights([embedding_matrix])
#  🔥 do NOT freeze; embedding layer can be trained! 🔥 ONLY part changed
noFreeze.layers[0].trainable = True



noFreeze.compile(
    optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
noFreeze_fit = noFreeze.fit(
    x = x_train_100, y = y_train,
    epochs = 15, batch_size = 32, validation_split = 0.2)
plt.plot([i+1 for i in range(15)],
         noFreeze_fit.history['acc'],
         label = "Training Accuracy")
plt.plot([i+1 for i in range(15)],
         noFreeze_fit.history['val_acc'],
         label = "Validation Accuracy")
plt.legend() ; plt.xlabel("Epoch") ; plt.ylabel("Accuracy")

'''
We observe a pretty BIG change here!

Training Accuracy pretty much reached PERFECT accuracy
Validation Accuracy also increases all the way up to 80%!
'''




















"""
This is the end of "GloVe" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""


















