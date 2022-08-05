#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:58:19 2022

@author: rj
"""

# https://www.tensorflow.org/guide/keras/functional




import os
os.chdir("/Users/rj/Desktop/RJstudio/V97")

from tensorflow.keras.backend import clear_session
clear_session()











"""
The Functional API 
 (▰˘◡˘▰)

According to  https://www.tensorflow.org/guide/keras/functional :

    "The Keras functional API is a way to create models that are more flexible 
    than the tf.keras.Sequential API." 🚀

All the videos I created BEFORE this video have used the Sequential API.
So, this video is an introductory video on the Functional API.


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""

# =============================================================================
# Importing
# =============================================================================

import numpy as np

# tensorflow below
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

















# =============================================================================
# Load MNIST model & Preprocess Data
# =============================================================================
'''
This part is exactly the SAME for BOTH 
Functional API & Sequential API

Since I already explained Preprocessing MNIST in a previous video,
I will SKIP detailed explanation (Link to previous video in Description)
'''

# load MNIST data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# REshape & REscale
x_train = x_train.reshape(60000, 784).astype("float32") / 255 # train
x_test  = x_test.reshape(10000, 784).astype("float32")  / 255 # test




















# =============================================================================
# Input Node
# =============================================================================

# create an input node
inputs = keras.Input(shape=(784,)) # REshaped earlier for this step


# check shape and dtype of inputs
inputs.shape # TensorShape([None, 784])
inputs.dtype # tf.float32






















# =============================================================================
# Graph of Layers
# =============================================================================

# Dense layer
dense = layers.Dense(units      = 64, 
                     activation = "relu")

# Pass inputs from earlier to the Dense layer
x = dense(inputs) # x as the output

# Another Dense layer to update x
x = layers.Dense(units      = 64,
                 activation = "relu")(x)

# Finally, outputs layer
outputs = layers.Dense(units      = 10,
                       activation = 'softmax')(x)
# 10 units since 10 categories to predict from MNIST dataset
    # 10 categories: 0, 1, 2, ... , 9





















# =============================================================================
# Create a Model
# =============================================================================

# use keras.Model() ; specify inputs & outputs from earlier
model = keras.Model(
    inputs  = inputs,  # remember: inputs = keras.Input(shape=(784,))
    outputs = outputs, # from just earlier
    name    = "simple_MNIST_model"
    )

# summary() of model to see all the layers
model.summary()



















# =============================================================================
# Compile & Fit
# =============================================================================
'''
This part is also identical to Sequential API! 

So, we use compile() & fit()
'''

# compilation step
model.compile(
    optimizer = keras.optimizers.RMSprop(),
    loss      = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics   = ["accuracy"]
    )
# Notice for Loss, SparseCategoricalCrossentropy is used
# because we have integer labels as opposed to One-Hot encoded labels
# I did NOT One-Hot encode labels / categories in this video 🚨


# fit model using training data
fit_model = model.fit(
    x                = x_train,   # training data
    y                = y_train,
    batch_size       = 64,
    epochs           = 10,
    validation_split = 0.3        # fraction used as validation data
    )













# =============================================================================
# Visualize Result & Evaluate
# =============================================================================

# Visualize Training & Validation Accuracy
import matplotlib.pyplot as plt

plt.plot([1,2,3,4,5,6,7,8,9,10],
         fit_model.history['accuracy'],
         label = "Train Accuracy")
plt.plot([1,2,3,4,5,6,7,8,9,10],
         fit_model.history['val_accuracy'],
         label = "Validation Accuracy")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
'''
Train Accuracy monotonically increases as expected

Validation Accuracy stops improving at around  96%  Accuracy
'''


# Accuracy on Test data using evaluate()
test_scores = model.evaluate(x       = x_test,  # test data
                             y       = y_test, 
                             verbose = 2)

# print Test Accuracy
print("Test Accuracy:", test_scores[1]) # 0.9674000144004822















# =============================================================================
# Save our model
# =============================================================================

# save the model using save() ;  put directory as argument
model.save("./funcAPImodel/")


# recreate the model by giving the same directory inside load_model()
model_recreated = keras.models.load_model("./funcAPImodel/")


# Check if we see the SAME architecture
model_recreated.summary()   # RECREATED model
model.summary()             # Original  model
# please examine Console outputs to check for the same architectures


# Evaluate with Test data to get Accuracy using RECREATED model
model_recreated.evaluate(x       = x_test, 
                         y       = y_test, 
                         verbose = 2)[1]
# SAME accuracy we got using Original model, good ✅
















"""
This is the end of "The Functional API" video~



Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""


















