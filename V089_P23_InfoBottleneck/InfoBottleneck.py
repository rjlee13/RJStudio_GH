#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 15:16:27 2022

@author: rj
"""


from tensorflow.keras.backend import clear_session
clear_session()

from tensorflow.keras.utils import set_random_seed
set_random_seed(7)









"""
Information Bottleneck
 (▰˘◡˘▰)

In this video, I want to demonstrate the IMPORTANCE🔥 of 
having sufficiently LARGE intermediate layers. 


Number of output classes of "Reuters" dataset is 46 
In other words, the model needs to learn to separate 46 different classes.

Therefore, if we insert an intermediate layer with 
the number of hidden units much 🚨FEWER🚨 than 46, then
you may end up PERMANENTLY dropping relevant information!! 🙀

We call this, "Information Bottleneck".

-----

To demonstrate Information Bottleneck right away,
I SKIP codes for loading / pre-processing "Reuters" dataset.

However, if you are interested in seeing entire code,
it is available at my Github page (Link provided in description)


A lot of explanation in this video is from a book titled,
"Deep Learning with Python" (Chapter 3)


Please 🌟PAUSE🌟 the video any time you want to understand code / output.
"""

























# =============================================================================
# Reuters dataset
# =============================================================================

# import Reuters dataset
from tensorflow.keras.datasets import reuters


# argument num_words=10000 restricts the data to 
# 10000 most frequently occurring words found in the data
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    num_words=10000)


# 8982 training examples
train_data.shape   # (8982,)
train_labels.shape # (8982,)


# to check if word number indices in data does NOT exceed 10000
flat_list = []
for xs in train_data:
    for x in xs:
        flat_list.append(x)  # flattening train_data list
max(flat_list) # 9999 < 10000 , max of FLATTENED list is 9999, good


# 2246 test examples
test_data.shape    # (2246,)
test_labels.shape  # (2246,)














# =============================================================================
# Preparing data
# =============================================================================

import numpy as np

# check outputs of enumerate() & np.zeros() below to understand what they do
for i, j in enumerate(train_data):
    print(i)
    print(j)

np.zeros((2,3))



# vectorize ; turn data into tensors
def vectorize_seq(seq, dimension = 10000):
    result = np.zeros((len(seq), dimension)) # create all-zero matrix
    for i, j in enumerate(seq):
        result[i, j] = 1   # set specific indices of result to 1
    return result

train_vector = vectorize_seq(train_data)
test_vector = vectorize_seq(test_data)



set(sorted(train_data[0])[:30])
# {1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15}
train_vector[0][:16]
# [0., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1.]
# you see 1. at the indices at {1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15} position
# otherwise 0.













# =============================================================================
# One-Hot Encode labels
# =============================================================================

from tensorflow.keras.utils import to_categorical


train_label_OHE = to_categorical(train_labels)
test_label_OHE  = to_categorical(test_labels)


# check
train_label_OHE[0]       # all zeros except for one element
train_label_OHE[0].shape # (46,)  46 elements
train_label_OHE[0].sum() # 1.0    only one of them is 1, the rest 0















# =============================================================================
# GOOD Model - NOT suffering from Information Bottleneck
# =============================================================================

# import what I need :) 
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

# build network / architecture of GOOD model
network = models.Sequential()
network.add(layers.Dense(units       = 64,    # SUFFICIENT number of units
                         activation  = 'relu',
                         input_shape = (10000,)))
network.add(layers.Dense(units       = 64,    # SUFFICIENT number of units
                         activation  = 'relu'))
network.add(layers.Dense(units       = 46,
                         activation  = 'softmax'))

# Compilation step
network.compile(optimizer = 'rmsprop',
                loss      = 'categorical_crossentropy',
                metrics   = ['accuracy'])

# Start fitting with Reuters dataset
fitting = network.fit(
    x                = train_vector,
    y                = train_label_OHE,
    epochs           = 13,
    batch_size       = 512,
    validation_split = 0.2 # fraction of train data used as validation data
    )

# Let me plot accuracy result
plt.plot(range(1, 14),
         fitting.history['accuracy'],
         label = 'Training Accuracy')
plt.plot(range(1, 14),
         fitting.history['val_accuracy'],
         label = 'Validation Accuracy')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

'''
Notice Validation Accuracy is around 80% at 13th Epoch

Now, let's see what happens when we have Information Bottleneck!
'''













network.summary()
clear_session()











# =============================================================================
# POOR model - suffering from Information Bottleneck  🙀
# I only change ONE part
# =============================================================================

# build network / architecture of POOR model
bottleneck = models.Sequential()
bottleneck.add(layers.Dense(units       = 64,
                            activation  = 'relu',
                            input_shape = (10000,)))
bottleneck.add(layers.Dense(units       = 2, # <- 🚨TOO FEW units🚨 ONLY change
                            activation  = 'relu'))
bottleneck.add(layers.Dense(units       = 46,
                            activation  = 'softmax'))

# Compilation step
bottleneck.compile(optimizer = 'rmsprop',
                   loss      = 'categorical_crossentropy',
                   metrics   = ['accuracy'])

# Start fitting with Reuters dataset
bottleneck_fit = bottleneck.fit(
    x                = train_vector,
    y                = train_label_OHE,
    epochs           = 13,
    batch_size       = 512,
    validation_split = 0.2 # fraction of train data used as validation data
    )

# Let me plot accuracy result
plt.plot(range(1, 14),
         bottleneck_fit.history['accuracy'],
         label = 'Training accuracy')
plt.plot(range(1, 14),
         bottleneck_fit.history['val_accuracy'],
         label = 'Validation accuracy')
plt.title('Accuracy with Information Bottleneck')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


'''
Notice Validation Accuracy is around  55% at 13th Epoch
'''











# bottleneck.summary()
# clear_session()











# =============================================================================
# Validation Accuracy Comparison
# =============================================================================

# Plot Validation Accuracies of GOOD & POOR models together
plt.plot(range(1, 14),
         fitting.history['val_accuracy'],         # GOOD model val acc
         label = 'GOOD model')
plt.plot(range(1, 14),
         bottleneck_fit.history['val_accuracy'],  # POOR model val acc
         label = 'POOR model - Info Bottleneck')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


'''
Clearly, we see drop in validation accuracy
when Information Bottleneck is happening.

The drop happens because the POOR model tried to compress 
a lot of information into the intermediate layer with ONLY 2 units. 

Some key information is lost during 'fit' process.
Once it is lost, it can NEVER be recovered by later layers ⚡⚡
'''




















"""
This is the end of "Information Bottleneck" video~


So we need sufficient number of units to avoid Information Bottleneck.

But we canNOT increase the number of units too much either~
Too many units makes the nework computationally expensive AND 
it may lead to learning UNwanted patterns!!!



"As always, deep learning is more an art than a science."



Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""
















