#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 17:44:00 2022

@author: rj
"""






from tensorflow.keras.backend import clear_session
clear_session()

from tensorflow.keras.utils import set_random_seed
set_random_seed(3)











"""
Overfitting in Deep Learning...
 (▰˘◡˘▰)

To prevent overfitting, one very good course of action is to
simply get MORE training data. 
However, that is sometimes very difficult (or impossible).😟


There are few others ways to prevent models from overfitting too much.
I want to show 3 common ways to avoid overfit for Deep Learning models  🔥
    1) Reducing Model Capacity
    2) L1 / L2 Regularization
    3) Dropout

To show the above 3 commons ways right away,
I 🚨SKIP🚨 data loading / preprocessing codes in the video. 
However, the entire code is available in my GitHub page. 



A lot of explanation in this video is from a book titled,
"Deep Learning with Python" (Chapter 4)


Please 🌟PAUSE🌟 the video any time you want to understand code / output.
"""














# =============================================================================
# Load IMDB dataset
# =============================================================================

from tensorflow.keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# we get 9999 due to `num_words=10000` when loading data
max([max(sequence) for sequence in x_train])  # 9999






# =============================================================================
# Preprocess X & Y
# =============================================================================

# to understand how enumerate() works
for i, j in enumerate(x_train):
    print(i, j)

import numpy as np
def vectorize_seq(sequence, dimension = 10000):
    results = np.zeros((len(sequence), dimension))
    for i, sequence in enumerate(sequence):
        results[i, sequence] = 1.
    return results


# vectorize x
x_train = vectorize_seq(x_train)
x_test = vectorize_seq(x_test)
# x shape
x_train.shape  # (25000, 10000) ; 25000 samples
x_test.shape   # (25000, 10000)


# vectorize y
y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')
# y shape
y_train.shape # (25000,)
y_test.shape  # (25000,)

















# =============================================================================
# ORIGINAL model - our baseline 🌟
# This is our starting point. We will make changes to this model later
# to experiment overfitting counter-measures!
# =============================================================================

# import what I need :) 
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers


# build architecture / network called 'original'
original = models.Sequential()
original.add(layers.Dense(units       = 16,
                          activation  ='relu',
                          input_shape = (x_train.shape[1],)))
original.add(layers.Dense(units       = 16,
                          activation  = 'relu'))
original.add(layers.Dense(units       = 1,
                          activation  = 'sigmoid'))

# compilation step
original.compile(optimizer ='rmsprop',
                 loss      ='binary_crossentropy',
                 metrics   = ['accuracy'])

# fit
fit_original = original.fit(
    x                = x_train,
    y                = y_train,
    epochs           = 17,
    batch_size       = 512,
    validation_split = 0.4
    )

# visualize training and validation loss
plt.plot(range(1, 18),
         fit_original.history['val_loss'],
         label = "ORIGINAL val loss")
plt.plot(range(1, 18),
         fit_original.history['loss'],
         label = "ORIGINAL training loss")
plt.xticks(range(1, 18))
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')

'''
As we expect, training loss is monotonically decreasing.

Validation loss, on the other hand, starts to INCREASE
as soon as after 3rd or 4th epoch!

It's reasonable to suspect that overfitting already started at 4th epoch.
'''
















# =============================================================================
# BIGGER model... Increase Model Capacity... 
# =============================================================================
'''
Let me INCREASE the number of units per layer
and give the model even MORE memorization capacity.

Everything else is the SAME as the original model

This way, overfitting happens even faster
as model can learn even MORE unwanted patterns... getting worse.. 🙃
I just wanted to demonstrate this to you :) 
'''

# build architecture, called bigger
bigger = models.Sequential()
bigger.add(layers.Dense(units       = 512,  # original was 16; CHANGED🚀
                        activation  ='relu',
                        input_shape = (x_train.shape[1],)))
bigger.add(layers.Dense(units       = 512,  # original was 16; CHANGED🚀
                        activation  = 'relu'))
bigger.add(layers.Dense(units       = 1,
                        activation  = 'sigmoid'))

# compilation step
bigger.compile(optimizer ='rmsprop',
               loss      ='binary_crossentropy',
               metrics   = ['accuracy'])

# fit
fit_bigger = bigger.fit(
    x                = x_train,
    y                = y_train,
    epochs           = 17,
    batch_size       = 512,
    validation_split = 0.4
    )

# visualize validation loss: ORIGINAL vs BIGGER
plt.plot(range(1, 18),
         fit_original.history['val_loss'],
         label = "ORIGINAL val loss")
plt.plot(range(1, 18),
         fit_bigger.history['val_loss'],
         label = "BIGGER val loss")
plt.xticks(range(1, 18))
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')

'''
As you can see, the BIGGER capacity model starts to overfit
very quickly.

AND, the BIGGER model's loss increases at a much faster rate than
that of ORIGINAL model.
'''















# =============================================================================
# SMALLER model
# =============================================================================
'''
This time, let me "reduce" the number of units to 4

We should see overfitting starting at a later epoch
AND overfitting will also occur at a slower pace.
'''

# build architecture 
smaller = models.Sequential()
smaller.add(layers.Dense(units       = 4,  # original was 16
                         activation  ='relu',
                         input_shape = (x_train.shape[1],)))
smaller.add(layers.Dense(units       = 4,  # original was 16
                         activation  = 'relu'))
smaller.add(layers.Dense(units       = 1,
                         activation  = 'sigmoid'))

# compilation step
smaller.compile(optimizer ='rmsprop',
                loss      ='binary_crossentropy',
                metrics   = ['accuracy'])

# fit
fit_smaller = smaller.fit(
    x                = x_train,
    y                = y_train,
    epochs           = 17,
    batch_size       = 512,
    validation_split = 0.4
    )

# visualize validation loss: ORIGINAL vs BIGGER vs SMALLER
plt.plot(range(1, 18),
         fit_original.history['val_loss'],
         label = "ORIGINAL val loss")
plt.plot(range(1, 18),
         fit_bigger.history['val_loss'],
         label = "BIGGER val loss")
plt.plot(range(1, 18),
         fit_smaller.history['val_loss'],
         label = "SMALLER val loss")
plt.xticks(range(1, 18))
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')

'''
As you can see validation loss for SMALLER model is decreasing
till about 7th epoch. 
And then it starts to increase again, overfitting.

Overfitting starts much later, and also overfitting happens at a 
much SLOWER rate compared to ORIGINAL and BIGGER models.

So, "model capacity" should be considered to diminish overfitting!🔥
'''














# =============================================================================
# Weight Regularization 
# =============================================================================
'''
This is another popular way to mitigate overfitting~!

"This method puts constraints on the complexity of a network 
by FORCING its weights to take 'small' values, which makes the
distribution of weight values more REGULAR."

The commoly used methods are called L1 & L2 regularization.
I will show L2 regularization effect.
'''

# import regularizer!
from tensorflow.keras import regularizers

# build architecture 
regular = models.Sequential()
regular.add(layers.Dense(units       = 16, # same as original model
                         kernel_regularizer = regularizers.l2(0.001), # 🚀
                         activation  ='relu',
                         input_shape = (x_train.shape[1],)))
regular.add(layers.Dense(units       = 16,  
                         kernel_regularizer = regularizers.l2(0.001), # 🚀
                         activation  = 'relu'))
regular.add(layers.Dense(units       = 1,
                         activation  = 'sigmoid'))

# compilation step
regular.compile(optimizer ='rmsprop',
                loss      ='binary_crossentropy',
                metrics   = ['accuracy'])

# fit
fit_regular = regular.fit(
    x                = x_train,
    y                = y_train,
    epochs           = 17,
    batch_size       = 512,
    validation_split = 0.4
    )

# visualize validation loss: ORIGINAL vs REGULARIZER
plt.plot(range(1, 18),
         fit_original.history['val_loss'],
         label = "ORIGINAL val loss")
plt.plot(range(1, 18),
         fit_regular.history['val_loss'],
         label = "REGULARIZER val loss")
plt.xticks(range(1, 18))
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')

'''
We see an interesting visualization result here.

The minimum value of REGULARIZER model's loss is greater than 
the ORIGINAL model's minimum loss value. (Can you think of why? 🤔)

However, REGULARIZER's model is much more resistant to overfitting.
In other words, loss increases at a much slower rate than ORIGINAL loss.
'''


















# =============================================================================
# Dropout
# =============================================================================
'''
Another common & effective way to alleviate overfitting is
Dropout.

"Dropout, applied to a layer, consists of randomly dropping out
(setting to 0) a number of output features of the layer during training.
... 
introducing noise in the output values of a layer 
can break up happenstance patterns that are NOT significant,
which the network will start memorizing if no noise is present."
'''

# build architecture 
dropout = models.Sequential()
dropout.add(layers.Dense(units       = 16,
                         activation  ='relu',
                         input_shape = (x_train.shape[1],)))
dropout.add(layers.Dropout(rate = 0.5))  # 50% of features ZEROED out 🚀
dropout.add(layers.Dense(units       = 16, 
                         activation  = 'relu'))
dropout.add(layers.Dropout(rate = 0.5))  # 50% of features DROPPED out 🚀
dropout.add(layers.Dense(units       = 1,
                         activation  = 'sigmoid'))

# compilation step
dropout.compile(optimizer ='rmsprop',
                loss      ='binary_crossentropy',
                metrics   = ['accuracy'])

# fit
fit_dropout = dropout.fit(
    x                = x_train,
    y                = y_train,
    epochs           = 17,
    batch_size       = 512,
    validation_split = 0.4
    )

# visualize validation loss: ORIGINAL vs DROPOUT
plt.plot(range(1, 18),
         fit_original.history['val_loss'],
         label = "ORIGINAL val loss")
plt.plot(range(1, 18),
         fit_dropout.history['val_loss'],
         label = "DROPOUT val loss")
plt.xticks(range(1, 18))
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')

'''
As you can see, 

DROPOUT model starts to overfit at a later epoch 
and the rate of increase of Loss is slower
than the ORIGINAL model
'''





















# =============================================================================
# All Validation Losses together 🏆
# =============================================================================
'''
I have shown 3 common ways to prevent overfitting in neural networks
    1) Smaller model capacity
    2) Weight regularization (L2 in particular)
    3) Dropout (with 50% rate)

Let me put all Validation Loss results together for comparison :)
'''

plt.plot(range(1, 18),
         fit_original.history['val_loss'],
         label = "ORIGINAL val loss")
plt.plot(range(1, 18),
         fit_bigger.history['val_loss'],
         label = "BIGGER val loss")
plt.plot(range(1, 18),
         fit_smaller.history['val_loss'],
         label = "SMALLER val loss")
plt.plot(range(1, 18),
         fit_regular.history['val_loss'],
         label = "REGULARIZER val loss")
plt.plot(range(1, 18),
         fit_dropout.history['val_loss'],
         label = "DROPOUT val loss")
plt.xticks(range(1, 18))
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')

# please PAUSE if you want to examine the visualization more :) 

















"""
This is the end of "Overfitting" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""

















