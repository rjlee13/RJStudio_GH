#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 14:04:21 2022

@author: rj
"""
import numpy as np
from tensorflow.keras.backend import clear_session
clear_session()















"""
Cross Validation Fit in Deep Learning
 (▰˘◡˘▰)

"When there is little data available, 
using K-fold validation is a great way to reliably evaluate a model."



`boston_housing` 🏘️ dataset only has
around 400 samples in the training set. 

Therefore, I will be conducting 5-fold Cross Validation 
to evaluate my small Deep Learning model~

I assume you understand how Cross Validation works, and 
we will be coding from scratch instead of relying on some modules/packages
to perform Cross Validation 🔥



A lot of explanation in this video is from a book titled,
"Deep Learning with Python" (Chapter 3)


Please 🌟PAUSE🌟 the video any time you want to understand code / output.
"""

















# =============================================================================
# Boston Housing Price dataset 🏘️
# =============================================================================

# Load dataset
from tensorflow.keras.datasets import boston_housing
(train_data, train_target), (test_data, test_target) = boston_housing.load_data()


# check dimensions of Boston Housing dataset

# train
train_data.shape   # (404, 13) <- only 404 samples in training set...
train_target.shape # (404,)

# test
test_data.shape    # (102, 13) <- 102 samples in test set
test_target.shape  # (102,)

'''
BTW,
The targets are the median values of owner-occupied homes, 
in thousands of dollars 💸
'''















# =============================================================================
# Feature-wise Normalization
# =============================================================================

# some features are on a much larger scale than the other
train_data[:,3].max()  # 1.0
train_data[:,7].max()  # 10.7103
train_data[:,9].max()  # 711.0
# some feature's max value is greater than 700
# while other feature's max is 1.0
# to make learning easier, 🚀 standard normalize our features!


# mean for each feature (column)
feature_mean = train_data.mean(axis = 0)
feature_mean
# quick check
train_data[:,0].mean()  # mean for first feature
feature_mean[0]         # get ~3.745 both times, good


# standard deviation for each feature
feature_std = train_data.std(axis = 0)
feature_std
# quick check
train_data[:,0].std()   # std for first feature
feature_std[0]          # get ~9.229 both times, good


# Standard Normalize train data
x_train = train_data - feature_mean # centering, make mean 0
x_train = x_train / feature_std     # standardizing, make std 1
# quick check for mean 0 std 1
x_train[:,0].mean()  # essentially 0 
x_train[:,0].std()   # essentially 1
x_train[:,7].mean()  # essentially 0
x_train[:,7].std()   # essentially 1


# Similarly, Standard Normalize test data
x_test = test_data - feature_mean # centering
x_test = x_test / feature_std     # standardizing
'''
Note that the quantities used for normalizing the "test" data 
are computed using the "training" data:
    feature_mean & feature_std are computed using 'Train' data NOT Test
'''














# =============================================================================
# Build Network / Architecture & Compilation
# =============================================================================

# Only 404 samples in train data as we saw earlier
x_train.shape
# with a small data, 
# our network should also be "small" to avoid overfitting 🚀

# import what I need :)
from tensorflow.keras import models, layers


# I will create a function that contains network/architecture & compilation
def network_compiler():
    
    # start building network
    network = models.Sequential()
    
    # input layer
    network.add(layers.Dense(units       = 64,
                             activation  = 'relu',
                             input_shape = (x_train.shape[1],)))
    
    # intermediate layer -- ONLY 1 intermediate layer to keep network small
    network.add(layers.Dense(units      = 64,
                             activation = 'relu'))
    
    # output layer
    '''
    network ends with a "single" unit and NO activation. 
    This is a typical setup for scalar regression
    '''
    network.add(layers.Dense(units = 1))
    
    # compilation step
    '''
    mse is a widely used loss function for regression problems.
    mae = mean absolute error
    '''
    network.compile(optimizer = 'rmsprop',
                    loss      = 'mse',
                    metrics   = ['mae'])
    
    # return network
    return network













# =============================================================================
# 5-fold Cross Validation Fit
# (please PAUSE to understand code if needed; I may be moving a bit fast)
# =============================================================================

# preparation for 5-fold CV
num_epoch = 100   # number of epochs when we fit model below
ks = 5            # 5-fold Cross Validation
fold_size = len(x_train) // ks
fold_size         # 80 ; so each fold has 80 samples (404/5 = ~80)
collect_MAE = []  # empty list for collecting all MAE metric as we train below



# for-loop to conduct 5-fold CV; so there will be 5 iterations
for k in range(ks):
    # so k ranges 0, 1, ... , 4 since ks = 5
    
    # print friendly messages telling us progress
    print(f'\nprocessing {k + 1}th fold\n')
    
    # generate validation data from x_train & train_target
    val_data   = x_train[k * fold_size: (k+1) * fold_size]
    val_target = train_target[k * fold_size: (k+1) * fold_size]
    
    # and then, REMAINING data automatically used for training / learning!
    x_train_remain = np.concatenate(          # remaining x_train
        [x_train[:k * fold_size],             # after validation data extracted
         x_train[(k+1) * fold_size:]],
        axis = 0)
    
    train_target_remain = np.concatenate(     # remaining train_target
        [train_target[:k * fold_size],        # after validation target extracted
         train_target[(k+1) * fold_size:]],
        axis = 0)
    
    # network/architecture: use the function created earlier
    network = network_compiler()
    
    # fit
    fit_result = network.fit(
        x               = x_train_remain,        # standard normalized data
        y               = train_target_remain,
        validation_data = (val_data, val_target),# provide validation data
        epochs          = num_epoch,
        batch_size      = 64
        )
    
    # append validation MAE for each fold iteration to collect_MAE
    collect_MAE.append(fit_result.history['val_mae'])
    # remember we created empty collect_MAE above
    
    
    
    
    
    












# clear_session()











# =============================================================================
# Analyzing MAEs collected
# =============================================================================

# Check all the MAEs collected
collect_MAE
# It is a list of lists. Notice double square-brackets: ]]
# Number of inner lists is the number of folds (= 5)
# Number of elements inside each inner list is number of epochs (= 100)


# double for-loop to find Mean MAE for each epoch (1st epoch to 100th epoch)
# Find the mean of FIVE 1st epochs, 2nd epochs, ... , 100th epochs each
# because we conducted FIVE-fold CV
MeanMAEperEpoch = [
    np.mean([mae[epoch] for mae in collect_MAE]) for epoch in range(num_epoch)]

MeanMAEperEpoch
MeanMAEperEpoch[0]  # mean Validation MAE for 1st   epoch was 21.216939544677736
MeanMAEperEpoch[99] # mean Validation MAE for 100th epoch was 2.505512237548828


# Visualize the change of Average MAE using matplotlib
import matplotlib.pyplot as plt
plt.plot(range(1, num_epoch + 1),
         MeanMAEperEpoch)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')


'''
MAE is decreasing fast until around 30th epoch.
Therefore, we probably want more than 30 epochs! 🚀

Validation MAE seems to stabilize at around 2.5 after around 60th epoch
'''


















"""
This is the end of "Cross Validation Fit" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""

















