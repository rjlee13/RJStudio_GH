#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:12:18 2022

@author: rj
"""


import numpy as np
from tensorflow.keras.backend import clear_session
clear_session()
















"""
Convnets: Convolutional Neural Network (CNN)
 (▰˘◡˘▰)

Convnets are very frequently used for various computer vision tasks 🚀

This video is an introduction to Convnets using Python,
so we will be classifying the famous MNIST digits~


A lot of explanation in this video is from a book titled,
"Deep Learning with Python" (Chapter 5)


I made a convnets video using R language some time ago.
If you are interested, please check out:
    https://youtu.be/qhGWCRO6trw
(Link in Description)


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""




















# =============================================================================
# Load MNIST data
# =============================================================================

# load data
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# check training set
train_images.shape # (60000, 28, 28); 60K images
train_labels.shape # (60000,)       ; 60K labels for 60K images
np.unique(train_labels, return_counts = True)
# 10 labels  : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# label count: [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949]
    # roughly 6000 count for each label


# check test set
test_images.shape # (10000, 28, 28); 10K images
test_labels.shape # (10000,)       ; 10K labels
np.unique(test_labels, return_counts = True)
# 10 labels  : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# label count: [ 980, 1135, 1032, 1010,  982,  892,  958, 1028,  974, 1009]
    # roughly 1000 count for each label

















# =============================================================================
# Preprocessing Data
# =============================================================================
'''
According to the textbook,
    "convnet takes as input tensors of shape 
    (image_height, image_width, image_channels)"
    
    
Because MNIST is a black-and-white picture --> image_channels == 1
image_channel represents "levels of gray" for MNIST
'''

# REshape & REscale training data
train_images = train_images.reshape((60000, 28, 28, 1)) # REshape
train_images = train_images.astype('float32') / 255     # REscale

# REshape & REscale test data
test_images = test_images.reshape((10000, 28, 28, 1))   # REshape
test_images = test_images.astype('float32') / 255       # REscale



# One-Hot Encoding labels; to_categorical() can do this for us
from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels  = to_categorical(test_labels)















# =============================================================================
# Convnet Architecture
# =============================================================================

# import what I need
from tensorflow.keras import layers, models


# let's start designing a simple Convnet architecture
cnn = models.Sequential()

'''
We create a stack of Conv2D & MaxPooling2D layers below

Notice the first Conv2D layer's input shape is (28,28,1)
This is the reason we REshaped earlier
'''
cnn.add(layers.Conv2D(filters     = 32,
                      kernel_size = (3,3),
                      activation  = 'relu',
                      input_shape = (28,28,1)))

cnn.add(layers.MaxPooling2D(pool_size = (2,2)))

cnn.add(layers.Conv2D(filters     = 64,
                      kernel_size = (3,3),
                      activation  = 'relu'))

cnn.add(layers.MaxPooling2D(pool_size = (2,2)))

cnn.add(layers.Conv2D(filters     = 64,
                      kernel_size = (3,3),
                      activation  = 'relu'))


'''
After a few Conv2D & MaxPooling2D layers above,
we Flatten() the 3D output to 1D
and then add few more Dense() layers like below
'''

cnn.add(layers.Flatten())
cnn.add(layers.Dense(units      = 64,
                     activation = 'relu'))
cnn.add(layers.Dense(units      = 10,
                     activation = 'softmax'))
# Output layer has 10 units with softmax activation
# because we are predicting 10 MNIST digit labels

# check our network so far
cnn.summary()




















# =============================================================================
# Compile & Fit
# =============================================================================

# Compilation step: set appropriate 1) optimizer 2) loss 3) metrics
cnn.compile(optimizer = 'rmsprop',
            loss      = 'categorical_crossentropy',
            metrics   = ['accuracy'])


# Fit using Training data
cnn_fit = cnn.fit(x          = train_images, # Train data
                  y          = train_labels, 
                  epochs     = 5, 
                  batch_size = 64)


# View Loss & Accuracy collected from fit() / training above
cnn_fit.history
# let me visualize the change of Accuracy data next






















# =============================================================================
# Visualize Training Accuracy
# =============================================================================

# import matplotlib
import matplotlib.pyplot as plt

# visualize
plt.plot([1,2,3,4,5],
         cnn_fit.history['accuracy'],
         label = "Accuracy")
plt.legend()
plt.xticks([1,2,3,4,5])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
'''
Training Accuracy increasing monotonically as we expect
'''



















# =============================================================================
# Evaluate on Test data
# =============================================================================

# use evaluate()
test_loss, test_acc = cnn.evaluate(x = test_images,  # Test data
                                   y = test_labels)

# Accuracy on Test set
test_acc # ~0.99   ;  Very Accurate!































"""
This is the end of "Intro to Convnets" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""

















