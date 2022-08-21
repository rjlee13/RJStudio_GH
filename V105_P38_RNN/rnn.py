#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 17:14:32 2022

@author: rj
"""


import matplotlib.pyplot as plt
from tensorflow.keras.backend import clear_session
clear_session()














"""
RNN: Recurrent Neural Networks
 (▰˘◡˘▰)


According to the textbook I am using:
    "RNN is a FOR loop that reuses quantities computed
    during the previous iteration of the loop, nothing more." 🚀


In this video, I show commonly used RNN techniques,
starting from SimpleRNN to LSTM, GRU, stacked GRU, and Bidirectional 🔥


I use IMDB movie datasets, which I preprocessed in previous videos too. 
So I 🚨SKIP🚨 preprocessing codes here. But...
If you are interested to see preprocessing code, it is available
at my GitHub page. 


A lot of explanation in this video is from a book titled,
"Deep Learning with Python" (Chapter 6)


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""



















# =============================================================================
# Import
# =============================================================================

# import what I need
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN, \
LSTM, GRU, Bidirectional


# =============================================================================
# Load & Preprocess IMDB data
# =============================================================================

# load IMDB data
from tensorflow.keras.datasets import imdb
# consider only 10K most common words
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 10000)


# preprocess
from tensorflow.keras import preprocessing

# Cut off reviews after 100 words by setting `maxlen = 100`
x_train = preprocessing.sequence.pad_sequences(sequences = x_train, 
                                               maxlen    = 80)
x_test  = preprocessing.sequence.pad_sequences(sequences = x_test, 
                                               maxlen    = 80)


















# =============================================================================
# SimpleRNN
# =============================================================================
'''
Simplest RNN; this is my baseline model with one Embedding & SimpleRNN

I want to see if more complex models (LSTM, GRU, etc) can surpass this 
Simplest RNN's performance. 
'''
# Architecture
simplestRnn = Sequential() 
simplestRnn.add(Embedding(input_dim  = 10000,
                          output_dim = 32))
simplestRnn.add(SimpleRNN(units      = 32))   # <- SimpleRNN 🐣
simplestRnn.add(Dense(units          = 1,
                      activation     = 'sigmoid'))
simplestRnn.summary()

# Compilation
simplestRnn.compile(
    optimizer = 'rmsprop',
    loss      = 'binary_crossentropy', # since binary problem
    metrics   = ['acc'])

# Fit
simplestRnn_fit = simplestRnn.fit(
    x                = x_train,
    y                = y_train,
    epochs           = 7,
    batch_size       = 128,
    validation_split = 0.3)

# Plot
plt.plot([i+1 for i in range(7)],
         simplestRnn_fit.history['acc'],
         label = "Training")
plt.plot([i+1 for i in range(7)],
         simplestRnn_fit.history['val_acc'],
         label = "Validation")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Validation Accuracy")

'''
Validation Accuracy reaches about 83% max with SimpleRNN.
'''




















# =============================================================================
# LSTM  (Long Short-Term Memory)
# =============================================================================
'''
According to the textbook:
    "
    ... SimpleRNN is generally too simplistic to be of real use.
    ... in practice, such long-term dependencies are impossible to learn 
    ... due to the 🎖️Vanishing Gradient Problem🎖️
    ...                ...
    ... LSTM and GRU layers are designed to solve this problem.
    ... (They are better at) saving information for later, thus preventing 
    older signals from gradually vanishing during processing.
    "
'''
# Architecture for LSTM
lstm = Sequential() 
lstm.add(Embedding(input_dim  = 10000,
                   output_dim = 32))
lstm.add(LSTM(units           = 32))   # <- LSTM 🐣
lstm.add(Dense(units          = 1,
               activation     = 'sigmoid'))
lstm.summary()

# Compilation
lstm.compile(
    optimizer = 'rmsprop',
    loss      = 'binary_crossentropy', # since binary problem
    metrics   = ['acc'])

# Fit
lstm_fit = lstm.fit(
    x                = x_train,
    y                = y_train,
    epochs           = 7,
    batch_size       = 128,
    validation_split = 0.3)

# Plot: SimpleRNN vs LSTM
plt.plot([i+1 for i in range(7)],
         lstm_fit.history['val_acc'],
         label = "LSTM")
plt.plot([i+1 for i in range(7)],
         simplestRnn_fit.history['val_acc'],
         label = "SimpleRNN")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Validation Accuracy")

'''
We can see that 
LSTM reaches a little higher validation accuracy than SimpleRNN model,
and LSTM reaches 85% accuracy max


Again, LSTM suffers much less from the Vanishing Gradient problem.
'''


















# =============================================================================
# GRU (Gated Recurrent Unit)
# =============================================================================
'''
According to the textbook:
    "
    Gated recurrent unit (GRU) layers work using the same principle as LSTM, 
    but they’re somewhat streamlined and thus cheaper to run (although they 
    may not have as much representational power as LSTM).
    "
'''
# Architecture for GRU
gru = Sequential() 
gru.add(Embedding(input_dim  = 10000,
                  output_dim = 32))
gru.add(GRU(units            = 32))   # <- GRU 🐣
gru.add(Dense(units          = 1,
              activation     = 'sigmoid'))
gru.summary()

# Compilation
gru.compile(
    optimizer = 'rmsprop',
    loss      = 'binary_crossentropy', # since binary problem
    metrics   = ['acc'])

# Fit
gru_fit = gru.fit(
    x                = x_train,
    y                = y_train,
    epochs           = 7,
    batch_size       = 128,
    validation_split = 0.3)

# Plot: SimpleRNN vs LSTM vs GRU
plt.plot([i+1 for i in range(7)],
         gru_fit.history['val_acc'],
         label = "GRU")
plt.plot([i+1 for i in range(7)],
         lstm_fit.history['val_acc'],
         label = "LSTM")
plt.plot([i+1 for i in range(7)],
         simplestRnn_fit.history['val_acc'],
         label = "SimpleRNN")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Validation Accuracy")

'''
GRU achieves slightly higher validation accuracy than SimpleRNN,
similar performance as the LSTM model.
'''















# =============================================================================
# Stacking Recurrent Layers  (2 GRU layers)
# =============================================================================
'''
According to the textbook:
    "
    Stacking Recurrent Layer is a classic way to build more powerful recurrent 
    networks: for instance, what currently powers the Google Translate 
    algorithm is a stack of 7 large LSTM! 🚀
    ...
    Stacked RNNs provide more representational power than a single RNN layer.
    "
                                    AND
    "
    To stack recurrent layers on top of each other in Keras, all intermediate 
    layers should return their full sequence of outputs
    "
'''
# Architecture
gru2 = Sequential() 
gru2.add(Embedding(input_dim   = 10000,
                   output_dim  = 32))
gru2.add(GRU(units             = 32,       # <- GRU #1 🐣
             return_sequences  = True))    # return full sequence of outputs
gru2.add(GRU(units             = 32))      # <- GRU #2 🐓
gru2.add(Dense(units           = 1,
               activation      = 'sigmoid'))
gru2.summary()

# Compilation
gru2.compile(
    optimizer = 'rmsprop',
    loss      = 'binary_crossentropy', # since binary problem
    metrics   = ['acc'])

# Fit
gru2_fit = gru2.fit(
    x                = x_train,
    y                = y_train,
    epochs           = 7,
    batch_size       = 128,
    validation_split = 0.3)

# Plot: LSTM vs GRU vs GRU * 2 
plt.plot([i+1 for i in range(7)],
         gru2_fit.history['val_acc'],
         label = "GRU*2")
plt.plot([i+1 for i in range(7)],
         gru_fit.history['val_acc'],
         label = "GRU")
plt.plot([i+1 for i in range(7)],
         lstm_fit.history['val_acc'],
         label = "LSTM")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Validation Accuracy")
'''
The visualization tells us that stacking 2 GRU layers does NOT give us much
performance improvement at all.

When preprocessing data (code available in GitHub, not from video), I CUT OFF
reviews after only 80 words to make training relatively fast. 

Maybe, the data is not complex enough to benefit from having 2 GRU layers.
It just made training time longer, without performance improvement
'''



















# =============================================================================
# Dropout
# =============================================================================
'''
According to the textbook:
    "
    Every recurrent layer in Keras has two dropout-related arguments: 
    
    1) dropout, a float specifying dropout rate for input units of layer,
    2) recurrent_dropout, specifying dropout rate of recurrent units.
    "
'''
# Architecture
gru2_drop = Sequential() 
gru2_drop.add(Embedding(input_dim   = 10000,
                        output_dim  = 32))
gru2_drop.add(GRU(units             = 32,       # <- GRU #1 🐣
                  return_sequences  = True,
                  dropout           = 0.2,
                  recurrent_dropout = 0.2))   
gru2_drop.add(GRU(units             = 32,       # <- GRU #2 🐓
                  dropout           = 0.2,
                  recurrent_dropout = 0.2))      
gru2_drop.add(Dense(units           = 1,
               activation           = 'sigmoid'))
gru2_drop.summary()

# Compilation
gru2_drop.compile(
    optimizer = 'rmsprop',
    loss      = 'binary_crossentropy', # since binary problem
    metrics   = ['acc'])

# Fit
gru2_drop_fit = gru2_drop.fit(
    x                = x_train,
    y                = y_train,
    epochs           = 7,
    batch_size       = 128,
    validation_split = 0.3)


# Plot: SimpleRNN vs LSTM
plt.plot([i+1 for i in range(7)],
         gru2_drop_fit.history['val_acc'],
         label = "GRU*2 Dropout Validation")
plt.plot([i+1 for i in range(7)],
         gru2_fit.history['val_acc'],
         label = "GRU*2 Validation")
plt.plot([i+1 for i in range(7)],
         gru_fit.history['val_acc'],
         label = "GRU Validation")
plt.plot([i+1 for i in range(7)],
         lstm_fit.history['val_acc'],
         label = "LSTM Validation")
plt.plot([i+1 for i in range(7)],
         simplestRnn_fit.history['val_acc'],
         label = "SimpleRNN Validation")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Accuracy")












# =============================================================================
# Bidirectional RNN
# =============================================================================
'''
According to the textbook:
    "
    A Bidirectional RNN exploits the order sensitivity of RNNs
    ...
    a Bidirectional RNN can catch patterns that may be overlooked by
    uni-directional RNN
    "
'''
# Architecture
gru_bd = Sequential() 
gru_bd.add(Embedding(input_dim     = 10000,
                     output_dim    = 32)) 
gru_bd.add(Bidirectional(GRU(units = 32)))   # <- GRU Bidirectional 🐣
gru_bd.add(Dense(units             = 1,
                 activation        = 'sigmoid'))
gru_bd.summary()

# Compilation
gru_bd.compile(
    optimizer = 'rmsprop',
    loss      = 'binary_crossentropy', # since binary problem
    metrics   = ['acc'])

# Fit
gru_bd_fit = gru_bd.fit(
    x                = x_train,
    y                = y_train,
    epochs           = 7,
    batch_size       = 128,
    validation_split = 0.3)

# Plot: LSTM vs GRU vs GRU Bidirectional
plt.plot([i+1 for i in range(7)],
         gru_bd_fit.history['val_acc'],
         label = "GRU Bidirection")
plt.plot([i+1 for i in range(7)],
         gru_fit.history['val_acc'],
         label = "GRU")
plt.plot([i+1 for i in range(7)],
         lstm_fit.history['val_acc'],
         label = "LSTM")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Validation Accuracy")
'''
According to the visualization, 
Bidirectional does NOT help improve Validation Accuracy.

In conclusion,
When reviews are cut off after 80 words, 
it seems like an Embedding layer with a single LSTM / GRU layer is sufficient,
without making the model too complex.

LSTM and Bidirectional tend to show strength in more complex, NLP problems.
'''













plt.plot([i+1 for i in range(7)],
         gru_bd_fit.history['val_acc'],
         label = "GRU Bidirection")
plt.plot([i+1 for i in range(7)],
         gru2_fit.history['val_acc'],
         label = "GRU*2")
plt.plot([i+1 for i in range(7)],
         gru_fit.history['val_acc'],
         label = "GRU")
plt.plot([i+1 for i in range(7)],
         lstm_fit.history['val_acc'],
         label = "LSTM")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Validation Accuracy")











"""
This is the end of "RNN" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""


















