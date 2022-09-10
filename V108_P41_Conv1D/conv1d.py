#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 10:25:03 2022

@author: rj
"""
import numpy as np

from tensorflow.keras.backend import clear_session
clear_session()












"""
1D Convolutions
 (▰˘◡˘▰)


According to the textbook I am using:
    🚀 "The same properties that make ConvNets excel at computer vision 
    ALSO make them [1D ConvNets] highly relevant to sequence processing." 
    
    🚀 "1D ConvNets can be competitive with RNNs on certain sequence-processing
    problems, usually at a considerably CHEAPER computational cost." 



In this video, I perform binary text classification problem using 
1D Convolutions! 😊



A lot of explanation in this video is from a book titled,
"Deep Learning with Python" (Chapter 6)


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""















# =============================================================================
# IMDB Data 🎦
# =============================================================================

# import IMDB from tensorflow
from tensorflow.keras.datasets import imdb

# Load IMDB data 
# Consider only the top 10,000 common words in the dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 10000)



# quick check of Train data

x_train.shape # (25000,) -> 25K movie reviews
y_train.shape # (25000,) -> 25K positive or negative review classifications
np.unique(y_train, return_counts = True)
# Only 2 unique labels in y_train: 0 & 1
# 12500 0's & 12500 1's
# 0 for negative & 1 for positive review
# So, this is a BINARY classification problem




















# =============================================================================
# Preprocessing Data 
# =============================================================================

# import preprocessing tool!
from tensorflow.keras.preprocessing import sequence



# Cut off reviews after 500 words by setting `maxlen = 500`
x_train_500 = sequence.pad_sequences(x_train,      # for Training data
                                     maxlen = 500)

x_test_500  = sequence.pad_sequences(x_test,       # for Test data
                                     maxlen = 500) 


# check with 70th and 36th samples

len(x_train[70])     # 787   <-- BEFORE preprocessing
len(x_train_500[70]) # 500   <-- truncated to 500


len(x_train[36])     # 51    <-- BEFORE preprocessing
len(x_train_500[36]) # 500   <-- padded with 0s to 500




















### more personal check
for i in range(100):
    print(i, len(x_train[i]))

x_train_500[36]
x_train[36]



















# =============================================================================
# Architecture 🏛️
# =============================================================================
'''
According to the textbook I am using:
    
    "1D convnets are structured in the same way as their 2D counterparts"
    
                                    AND
    
    "One difference, though, is the fact that you can afford to use 
    LARGER convolution windows with 1D convnets""
'''
# import what I need
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


# build linear stack of layers
conv1d = Sequential()

# Embedding Layer
conv1d.add(layers.Embedding(input_dim    = 10000,
                            output_dim   = 128,
                            input_length = 500))


# 🌟  Conv1D & MaxPooling1D layers  🌟
conv1d.add(layers.Conv1D(filters     = 32,
                         kernel_size = 7,    # larger convolution window
                         activation  = 'relu'))
conv1d.add(layers.MaxPooling1D(pool_size = 5))


# Global Max Pooling + 1 Dense 
conv1d.add(layers.GlobalMaxPooling1D())
conv1d.add(layers.Dense(units = 1))


# check architecture
conv1d.summary()

















# =============================================================================
# Compile & Fit 🔨
# =============================================================================

# import optimizer
from tensorflow.keras.optimizers import RMSprop

# Compilation
conv1d.compile(
    optimizer = RMSprop(),             # RMSprop was imported above
    loss      = 'binary_crossentropy', # this is binary classification
    metrics   = ['acc'])               # monitor training accuracy

# Fit
epoch_num = 8
conv1d_fit = conv1d.fit(
    x                = x_train_500,    # Train data
    y                = y_train,
    epochs           = epoch_num,
    batch_size       = 128,
    validation_split = 0.2)


# Training History result (Accuracy & Loss)
conv1d_fit.history

# Let me visualize training result next!

















# =============================================================================
# Training & Validation Accuracies
# =============================================================================

# import matplotlib
import matplotlib.pyplot as plt


# Visualize
plt.plot([i+1 for i in range(epoch_num)],       # Training Accuracy
         conv1d_fit.history['acc'],
         label = "Training Acc")
plt.plot([i+1 for i in range(epoch_num)],       # Validation Accuracy
         conv1d_fit.history['val_acc'],
         label = "Validation Acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
'''
Training Accuracy is monotonically increasing as expected.

Validation Accuracy reaches around 89% accuracy max.
'''

















# =============================================================================
# Combine CNN & RNN
# =============================================================================
'''
According to the textbook I am using:
    
    "One strategy to combine the speed and lightness of ConvNet with 
    order-sensitivity of RNNs is to use a 1D ConvNet as a preprocessing step 
    before an RNN" 🔮
'''
# build linear stack of layers
cnn_rnn = Sequential()

# Embedding + CNN   ;   SAME as before
cnn_rnn.add(layers.Embedding(input_dim    = 10000,
                             output_dim   = 128,
                             input_length = 500))
cnn_rnn.add(layers.Conv1D(filters     = 32,
                          kernel_size = 7,
                          activation  = 'relu'))
cnn_rnn.add(layers.MaxPooling1D(pool_size = 5))


# RNN   ;   Recurrent layer (GRU)      NEWLY MODIFIED/ADDED 🌟
cnn_rnn.add(layers.GRU(units   = 32,
                       dropout = 0.2,
                       recurrent_dropout = 0.5,
                       return_sequences  = True))
cnn_rnn.add(layers.GRU(units   = 64,
                       dropout = 0.1,
                       recurrent_dropout = 0.2))

# output Dense layer, then Compile & Fit
cnn_rnn.add(layers.Dense(units = 1))
cnn_rnn.compile(optimizer = RMSprop(),
                loss      = 'binary_crossentropy',
                metrics   = ['acc'])
cnn_rnn_fit = cnn_rnn.fit(x = x_train_500, y = y_train,
                          epochs = epoch_num, batch_size = 128,
                          validation_split = 0.2)


# Plot Validation Accuracies:   ONLY Conv1D  vs  Conv1D + GRU
plt.plot([i+1 for i in range(epoch_num)],
         conv1d_fit.history['val_acc'],
         label = "ONLY Conv1D")
plt.plot([i+1 for i in range(epoch_num)],
         cnn_rnn_fit.history['val_acc'],
         label = "Conv1D + GRU")
plt.xlabel("Epochs"), plt.ylabel("Accuracy"), plt.legend()
plt.title("Validation Accuracy")


'''
Until 8th epoch, combining CNN with RNN only negatively influenced 
Validation Accuracy performance!  (Validation Acc decreased)

Plus, it took MUCH LONGER 🕰️ to train for each epoch.

According to the textbook 🚀:
    "[Combining CNN & RNN] is especially beneficial when you’re dealing with 
    sequences that are so long they can’t realistically be processed with RNNs, 
    such as sequences with thousands of steps."  

So, it seems like our binary problem was NOT complex enough to benefit
from combining CNN & RNN! 
'''





















"""
This is the end of "Conv1D" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""


















