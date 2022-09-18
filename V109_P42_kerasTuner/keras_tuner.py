#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 21:19:48 2022

@author: rj
"""

# conda install -c conda-forge keras-tuner


import numpy as np














"""
Keras Tuner: HyperTuning
 (▰˘◡˘▰)



This video shows you how to TUNE Neural Network model's HyperParameters  
using Keras Tuner 🚀 


In previous videos, I manually decided HyperParameters' values based on 
my understanding / experience. 

But ... Arbitrary hyperparameters probably return SUBOPTIMAL result! 🙅‍

That's why HyperParameter Tuning is a VERY important skill! 😊


-----


A lot of explanation in this video is from this link:
    https://www.tensorflow.org/tutorials/keras/keras_tuner
    (Link in Description)


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""















# =============================================================================
# Fashion-MNIST 👗👞👕 dataset
# =============================================================================

# import datasets, which contains Fashion-MNIST dataset
from tensorflow.keras import datasets


# Load Fashion-MNIST 👗👞👕
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()



# Quick glance at Training data:
x_train.shape # 60K images; 28 x 28 grayscale image
y_train.shape # 60K labels
np.unique(y_train, return_counts = True)
# 10 unique labels/values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# each label occurring 6K times


# Quick Visualization of Sample #99 Training image 
import matplotlib.pyplot as plt
plt.imshow(x_train[99], cmap = "Greys")
# it's a bag! 

















# =============================================================================
# Rescale & One Hot Encode 🛠️
# =============================================================================

# Rescale so that all values are within [0,1]; divide all by 255
x_train_rescale = x_train.astype('float32') / 255.0  # for Train data
x_test_rescale  = x_test.astype('float32')  / 255.0  # for Test  data


# Quick Check
x_train[1].max()         # 255; BEFORE Rescale
x_train_rescale[1].max() # 1.0; AFTER  Rescale



# import to_categorical to perform One Hot Encoding (OHE)
from tensorflow.keras.utils import to_categorical

# Perform OHE
y_train_OHE = to_categorical(y_train)
y_test_OHE  = to_categorical(y_test)


# Quick Check
y_train_OHE.shape # There are 10 columns since there are 10 unique labels
y_train_OHE       # there should be single 1 in each row, looks good
                  # (some 1's are hidden)




















# =============================================================================
# Hypermodel 🔮
# =============================================================================
'''
According to the link I am using:
    "The model you set up for hypertuning is called a hypermodel"

We create a function called model_builder() to compile a model which contains
information about the HyperParameters to be TUNED
'''

# import what I need
from  tensorflow.keras import models, layers, optimizers


# model_builder function!
def model_builder(hp): # hp for HyperParameter
    
    # build linear stack of layers sequentially, using `Sequential()`
    model = models.Sequential() # sequential model
    
    # FLATTEN images so that forthcoming DENSE Layers can process data
    model.add(layers.Flatten(input_shape = (28, 28)))
    
    '''
    Placeholder for TUNING number of units in DENSE layer
    specify Integer range of possible values to try, using `Int`
    '''
    hp_units = hp.Int('units', 
                      min_value = 32,   # lowest  value to try during TUNING
                      max_value = 512,  # highest value to try during TUNING
                      step      = 64) 
    
    '''
    Placeholder for TUNING the activation functions in DENSE layer
    specified 3 potential activation functions below, using `Choice`
    '''
    hp_activation = hp.Choice('activation',
                              values = ['relu', 'sigmoid', 'tanh'])
    
    '''
    DENSE layer
    specify units & activation need to be TUNED
    '''
    model.add(layers.Dense(units      = hp_units,
                           activation = hp_activation))
    
    # output layer
    model.add(layers.Dense(units      = 10, # 10, since 10 labels to predict
                           activation = 'softmax'))
    
    '''
    Compilation
    this function must return COMPILED model, so we `compile()` here
    '''
    model.compile(
        optimizer = optimizers.Adam(),     # Adam optimization
        loss = 'categorical_crossentropy', # since we have 10 labels to predict
        metrics = ['accuracy'])            # monitor Accuracy during Training
    
    
    # finally return COMPILED model
    return model





















# =============================================================================
# Hyperband Tuner & EarlyStopping  🐣
# =============================================================================

# import Keras Tuner
import keras_tuner as kt

'''
According to the link I am using:
    
    "Hyperband tuning algorithm uses adaptive resource allocation 
    and early-stopping to quickly converge on a high-performing model."
    
                                AND
                    
    "[Hyperband] trains a large number of models for a few epochs and carries 
    forward only the top-performing half of models to the next round"

Note there are other tuners: RandomSearch, BayesianOptimization, Sklearn
'''

# we use Hyperband Tuner
tuner = kt.Hyperband(
    hypermodel   = model_builder,  # function created earlier
    objective    = 'val_accuracy', # want validation accuracy to be good
    max_epochs   = 7,              # max number of epochs to train one model
    
    # specify path / directory to save work by Hyperband tuner
    directory    = '/Users/rj/Desktop/RJstudio/V109',
    project_name = 'hyper-tuning'
    )



# Next, EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping

# because we train LOTS of models when tuning,
# it's probably a GOOD idea to set up EarlyStopping! 🌟

# We stop training if there is NO improvement in val_accuracy for 3 epochs
stop_early = EarlyStopping(
    monitor  = 'val_accuracy',  # Quantity to be monitored
    patience = 3            
    )

















# =============================================================================
# Tuning 🚀 HyperParameter Search
# =============================================================================

# start tuning! 
# following looks similar to the usual fit() function
tuner.search(                       # HyperParameter Search
    x_train_rescale,                # Rescaled Training images
    y_train_OHE,                    # One Hot Encoded labels
    epochs           = 20,
    validation_split = 0.2,
    callbacks        = [stop_early] # EarlyStopping created above
    )
# this takes some time!



# once Tuning is done,
# obtain BEST HyperParameters by `get_best_hyperparameters()`
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

# for example, 
best_hps.get('units')       # TUNED number of units
best_hps.get('activation')  # TUNED activation function

# view all TUNED values at once like this:
best_hps.values

















# =============================================================================
# Train model with BEST / TUNED HyperParameters  🫠
# =============================================================================

# build() model with optimal / tuned HyperParameters (best_hps)
hyper_model = tuner.hypermodel.build(best_hps) # best_hps from above

# Fit
hyper_model_fit = hyper_model.fit(
    x                = x_train_rescale, # Train data   
    y                = y_train_OHE, 
    epochs           = 25, 
    validation_split = 0.2,
    callbacks        = [stop_early]     # same EarlyStopping as before
    )



# View Training Result as a Pandas DataFrame
import pandas as pd
pd.DataFrame(hyper_model_fit.history)


# Next, let me visualize Training Result

















# =============================================================================
# Training & Validation Accuracies
# =============================================================================

# Training "Early-Stopped" at 11th epoch
epochs = 11


# Visualize Training Result
plt.plot([i+1 for i in range(epochs)],
         hyper_model_fit.history['accuracy'],
         label = "Training Acc")
plt.plot([i+1 for i in range(epochs)],
         hyper_model_fit.history['val_accuracy'],
         label = "Validation Acc")
plt.xlabel("Epochs"), plt.ylabel("Accuracy"), plt.legend()



'''
Training Accuracy increasing monotonically as expected

Validation Accuracy reaches ~ 89% max
'''


















# =============================================================================
# HyperModel Evaluation with Test Data 
# =============================================================================

# use `evaluate()`
evaluation = hyper_model.evaluate(x = x_test_rescale, # Test Data
                                  y = y_test_OHE)

# Test dataset accuracy
evaluation[1]  # ~ 88%





















"""
This is the end of "Keras Tuner" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""















