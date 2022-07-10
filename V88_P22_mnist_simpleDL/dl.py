#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 23:26:35 2022

@author: rj

conda install tensorflow -c conda-forge
 

This exmaple is often the very first exercise done by 
many people interested in learning Deep Learning


It is safe to use from tensorflow.keras. 
instead of from keras. while importing all the necessary modules.
"""
from tensorflow.keras.backend import clear_session
clear_session()















"""
MNIST: classify handwritten digits
 (▰˘◡˘▰)

MNIST stands for Modified National Institute of Standards and Technology 

We are classifying grayscale images of handwritten ✍️ digits
into 10 categories: 0, 1, 2, ... , 9

I created a simple function that shows a handwritten digit AND
displays what my Neural Network model predicts below! 🚀



A lot of explanation in this video is from a book titled,
"Deep Learning with Python" (Chapter 2)


Please 🌟PAUSE🌟 the video any time you want to understand code / output.
"""

def predict_digit(nth_sample):
    digit = test_images[nth_sample] 
    plt.imshow(digit, cmap = 'Greys')
    number = np.argmax(network.predict(test_images_reshape)[nth_sample])
    print(f'Model predicts as:  {number}')

# let's test with some samples    
predict_digit(333)


 

















# =============================================================================
# Load MNIST data
# =============================================================================

# modules used
import numpy as np
from tensorflow.keras.datasets import mnist # <- MNIST here :) 


# assign MNIST train & test data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# let's take a look at our data...

# Train data
train_images.shape # (60000, 28, 28): 60000 images, each 28x28 pixels
train_labels.shape # (60000,)         60000 categories of (0, 1, 2, .., 9)
np.unique(train_labels, return_counts = True)
# we see 10 categories (0, 1, 2, .., 9)
# each category appearing roughly 6000 times


# Test data
test_images.shape # (10000, 28, 28): 10000 images, each 28x28 pixels
test_labels.shape # (10000,)         10000 categories of (0, 1, 2, .., 9)
np.unique(test_labels, return_counts = True)
# we see 10 categories (0, 1, 2, .., 9)
# each category appearing roughly 1000 times













# =============================================================================
# REshape & REscale MNIST data
# =============================================================================

'''
We need to PRE-Process by REshaping our data
into the shape the neural network expects.

initial shape   (60000, 28, 28)
        ==REshape==
after REshape   (60000, 28 * 28)



We also want to REscale all the values 
so that they are in the [0,1] interval.
'''

# let's take a look at the 1st image BEFORE REshape
train_images[0].shape # (28, 28)
train_images[0]       # and we can see values ranging from 0 - 255


# REshape train data
train_images_reshape = train_images.reshape((60000, 28*28))
# REscale train data
train_images_reshape = train_images_reshape.astype('float32')/255

# REshape test data
test_images_reshape = test_images.reshape((10000, 28*28))
# REscale test data
test_images_reshape = test_images_reshape.astype('float32')/255


# let's take a look at the 1st image AFTER  REshape & REscale
train_images_reshape[0].shape # (784,) ; 784 = 28 * 28
train_images_reshape[0]       # now values ranging from 0 - 1

















# =============================================================================
# One-Hot Encode Train/Test Labels
# =============================================================================

# luckily, to_categorical() can do this for us in one command 🙌
from tensorflow.keras.utils import to_categorical

# One-Hot Encode
train_labels_OHE = to_categorical(train_labels)
test_labels_OHE  = to_categorical(test_labels)


# quick check
# see the first 5 rows
# check if we have 10 columns for 10 categories, and ONE 1 in each row
train_labels_OHE[:5] # good






















# =============================================================================
# build Network / Architecture 🛠️
# =============================================================================

# Need models & layers 
from tensorflow.keras import models, layers


# initiate network / architecture
network = models.Sequential()

# NOW, build a chain of 2 Dense / fully-connected layers

# relu dense 1st layer 
# with input_shape 28 * 28 <- this is why we REshaped earlier
network.add(layers.Dense(units       = 512, 
                         activation  = 'relu', 
                         input_shape = (28 * 28,)))

# softmax dense 2nd layer to return 10 probability scores, summing to 1
network.add(layers.Dense(units      = 10, 
                         activation = 'softmax'))
# 1 of 10 categories (0, 1, 2, .., 9) that receives 
# the HIGHEST probability will be chosen


#----- tips :)

# we set input shape to be 784 = 28 * 28, let's check it
network.input_shape # (None, 784) , good!

# we can review our network / architecure using summary()
network.summary()


















# =============================================================================
# Compilation Step
# =============================================================================

'''
Decide 3 more things:
    
    1) optimizer - mechanism through which the network will update itself
    based on the data it sees and its loss function
    In this video, rmsprop. Don't worry about what this exactly is for now.
    
    2) loss function - How the network will be able to measure its performance 
    on the training data
    In this video, categorical_crossentropy which measures the distance between
    the prob distritubion output by network and the true distribution
    
    3) metrics - what to monitor during training and testing
    In this video, we will monitor 'accuracy'
'''

network.compile(optimizer = 'rmsprop',
                loss      = 'categorical_crossentropy',
                metrics   = ['accuracy'])


network.optimizer
network.loss
network.metrics_names














# =============================================================================
# Train / Fit network 
# =============================================================================

# Train network using fit() method
fitting = network.fit(
    x          = train_images_reshape,  # REshaped REscaled data
    y          = train_labels_OHE,      # One Hot Encoded labels
    epochs     = 5,   # 5 iterations over all training data
    batch_size = 128) # training in mini-batches of 128 samples

# for each epoch, we can view loss & accuracy quantities

# we can also view ALL loss & accuracy with .history
fitting.history

# let me visualize loss & accuracy real quick next!



















# =============================================================================
# Plot Loss & Accuracy
# =============================================================================

# need matplotlib to plot
import matplotlib.pyplot as plt


# Accuracy plot
plt.plot([1,2,3,4,5],
         fitting.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.xticks([1,2,3,4,5])
# we see accuracy increasing, good!


# Loss plot
plt.plot([1,2,3,4,5],
         fitting.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xticks([1,2,3,4,5])
# we see loss decreasing, good!

















# =============================================================================
# Perfomance on Test set!
# =============================================================================

# Use network's evaluate() method
test_loss, test_acc = network.evaluate(test_images_reshape, test_labels_OHE)

print(f'Accuracy on Test data was {test_acc:.3f}')


















"""
This is the end of "MNIST" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""


















