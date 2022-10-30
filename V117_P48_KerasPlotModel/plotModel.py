#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 22:17:40 2022

@author: rj
"""



from tensorflow.keras.backend import clear_session
clear_session()











"""
Keras Plot Model
 (▰˘◡˘▰)


Keras has a very nice plotting function to help us examine
our Neural Network Model / Architecture / Layers very clearly 🚀

I have an example on the right 👉



To show how to plot Neural Network model ASAP, 
I SKIM through the code for PreProcessing MNIST datasets.

I basically re-use the simple ConvNet model from my previous YouTube video. 
If you want more code explanation, please check it out ~ 🐣
( https://youtu.be/A93WsiiyyZ8 )




A lot of explanation in this video is from a book titled,
"Deep Learning with Python" (Chapter 7)


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""




















# =============================================================================
# MNIST Data Preprocessing
# =============================================================================
'''
As I mentioned earlier,
I skim through PreProcessing steps/code!
'''

# load MNIST dataset
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# Reshape & Rescale image data
train_images = train_images.reshape(60000, 28, 28, 1).astype("float32") / 255
test_images  =  test_images.reshape(10000, 28, 28, 1).astype("float32") / 255


# One Hot Encode labels, using to_categorical
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels  = to_categorical(test_labels)


















# =============================================================================
# Simple CNN Architecture
# =============================================================================

# import models & layers
from tensorflow.keras import models, layers

# simple ConvNet Architecture
cnn = models.Sequential()

'''
Conv2D & MaxPooling2D layers below
'''
cnn.add(layers.Conv2D(filters     = 32,
                      kernel_size = (3,3),
                      activation  = 'relu',
                      input_shape = (28,28,1)))

cnn.add(layers.MaxPooling2D(pool_size = (2,2)))


cnn.add(layers.Conv2D(filters     = 64,
                      kernel_size = (3,3),
                      activation  = 'relu'))

'''
then Flatten, add Dense + Output layers
'''
cnn.add(layers.Flatten())
cnn.add(layers.Dense(units      = 64,
                     activation = 'relu'))
cnn.add(layers.Dense(units      = 10,
                     activation = 'softmax'))

# check our simple cnn architecture
cnn.summary()


# =============================================================================
# (quick) Sanity Test
# =============================================================================

# Compile
cnn.compile(optimizer = 'rmsprop',
            loss      = 'categorical_crossentropy',
            metrics   = ['accuracy'])

# Fit
cnn_fit = cnn.fit(x          = train_images, # Train data
                  y          = train_labels, 
                  epochs     = 5, 
                  batch_size = 64)

# Training Accuracy is reaching 99%; model IS learning
import pandas as pd
pd.DataFrame(cnn_fit.history)
# Sanity Test ✅














# =============================================================================
# Keras Plot Model 🚀
# =============================================================================

# import plot_model
from tensorflow.keras.utils import plot_model

'''
According to the textbook:
    "
    [plot_model] requires that you’ve installed the Python pydot and pydot-ng 
    libraries as well as the graphviz library.
    "


I used the following commands to install 2 libraries:
    conda install -c conda-forge pydot
    conda install -c anaconda graphviz
'''

# plot model
plot_model(cnn)


# can also display shape information
plot_model(cnn,
           show_shapes = True)

# save as PNG file
plot_model(cnn,
           show_shapes = True,
           to_file     = '/Users/rj/Desktop/RJstudio/V117/SimpleCnn.png')
























"""
This is the end of "Keras Plot Model" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""


















