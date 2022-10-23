#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 22:44:30 2022

@author: rj
"""
from tensorflow.keras.backend import clear_session
clear_session()








"""
TensorBoard
 (▰˘◡˘▰)


TensorBoard is an awesome BROWSER-based visualization tool 🚀
It tells you lots of information regarding all kinds of processes happening
during your neural network model training.




To show how to view TensorBoard ASAP, I will 'quickly' skim through the code 
for classifying MNIST digits. 
If you want more explanation regarding the code, 
please check out my previous YouTube video( https://youtu.be/oo6zHm7gUrc ) 🫶




-----


A lot of explanation in this video is from a book titled,
"Deep Learning with Python" (Chapter 7)


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""













# =============================================================================
# MNIST Data Preprocessing
# =============================================================================
'''
As I mentioned earlier,
I skim through code / PreProcessing steps!
'''

# load MNIST dataset
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Reshape & Rescale image data
x_train = x_train.reshape(60000, 28*28).astype("float32") / 255
x_test  = x_test.reshape(10000, 28*28).astype("float32") / 255


# One Hot Encode labels
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)
















# =============================================================================
# Neural Network Architecture
# =============================================================================

# import models & layers
from tensorflow.keras import models, layers

# Simple Neural Network Architecture
mnistModel = models.Sequential()
mnistModel.add(layers.Dense(units       = 512,
                            activation  = 'relu',
                            input_shape = (784,)))
mnistModel.add(layers.Dense(units      = 128,
                            activation = 'relu'))
mnistModel.add(layers.Dense(units      = 10,
                            activation = 'softmax'))

# Compilation
mnistModel.compile(
    optimizer = 'rmsprop',
    loss      = 'categorical_crossentropy',
    metrics   = ['acc'])


# summary of mnistModel
mnistModel.summary()












# =============================================================================
# TensorBoard
# =============================================================================

# import callbacks
from tensorflow.keras import callbacks

# TensorBoard Callback
tb = callbacks.TensorBoard(
    
    # place to store log files
    log_dir = '/Users/rj/Desktop/RJstudio/V116/tensorboard',
    
    # generate activation histogram for each epoch
    histogram_freq  = 1 
    )


# start training with TensorBoard callback
mnist_fit = mnistModel.fit(
    x = x_train,
    y = y_train,
    epochs = 7,
    validation_split = 0.2,
    callbacks = [tb]   # 🌟 TensorBoard here
    )


'''
Time to launch Tensorboard 🚀


from my COMMAND LINE, I execute:
    tensorboard --logdir=/Users/rj/Desktop/RJstudio/V116/tensorboard


and then browse to:
    http://localhost:6006
'''






















"""
This is the end of "TensorBoard" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""

















