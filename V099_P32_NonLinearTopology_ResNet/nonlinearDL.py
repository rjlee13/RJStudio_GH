# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np

from tensorflow.keras.backend import clear_session
clear_session()






















"""
Non-linear Connectivity Topology
 (▰˘◡˘▰)

According to https://www.tensorflow.org/guide/keras/functional#a_toy_resnet_model :
(Link in Description)

    "The functional API makes it easy to manipulate 
    NON-linear connectivity topologies
    
    these are models with layers that are NOT connected sequentially, 
    which the Sequential API canNOT handle." 🚀


This video builds a "toy" ResNet model, which has  
non-linear connectivity topology

Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""



















# =============================================================================
# CIFAR10 dataset
# =============================================================================

# load CIFAR10 dataset
from tensorflow import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
'''
To learn more about CIFAR10 dataset, please check out:
    https://www.cs.toronto.edu/~kriz/cifar.html
(Link in Description)

In particular, 
    "CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, 
    with 6000 images per class. 
    There are 50000 TRAINING images and 10000 TEST images."
'''

# Training data dimension
x_train.shape # (50000, 32, 32, 3); 50K 32x32 training images
y_train.shape # (50000, 1)        ; 50K labels for 50K images
np.unique(y_train, return_counts = True)
# 10 classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# 5000 count for each class



# Test data dimension
x_test.shape # (10000, 32, 32, 3); 10K 32x32 test images
y_test.shape # (10000, 1)        ; 10K labels
np.unique(y_test, return_counts = True)
# 10 classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# 1000 count for each class





















# =============================================================================
# Visualize 1 of Training Images (for fun 🫠)
# =============================================================================

# let me visualize image #7 just to check out CIFAR10 dataset real quick!
import matplotlib.pyplot as plt
plt.imshow(x_train[7])
# It is a horse! 🐴

'''
It is a 32x32 image. So the quality of the image is NOT good.

For example, the image becomes very blur if I enlarge it.
'''

























# =============================================================================
# Preprocess CIFAR10 data
# =============================================================================

# Each pixel is represented by 8 bits
# so the max value after flattening an image is 255 
x_train[0].flatten().max()  # 255
x_train[35].flatten().max() # 255


# Rescale so that all values are within [0,1]
x_train_rescale = x_train.astype("float32") / 255.0
x_test_rescale  = x_test.astype("float32")  / 255.0


# Now check the max values after Rescaling
x_train_rescale[0].flatten().max()  # 1.0  (as opposed to 255)
x_train_rescale[35].flatten().max() # 1.0  




# One-Hot Encoding classes; to_categorical() can do it for us
y_train_OHE = keras.utils.to_categorical(y_train, 10)
y_test_OHE  = keras.utils.to_categorical(y_test,  10)

# check OHE
y_train.shape     # (50000, 1)   ; before OHE only 1 column
y_train_OHE.shape # (50000, 10)  ; now we have 10 columns for 10 classes























# =============================================================================
# A "toy" ResNet Architecture
# =============================================================================

'''
I show architecture diagram from
https://www.tensorflow.org/guide/keras/functional#a_toy_resnet_model

Make sure to notice the NON-linear topology of the architecture
'''


# import layers
from tensorflow.keras import layers

# input node
inputs = keras.Input(shape = (32,32,3), 
                     name  = "cifar10img")



# block 1
x = layers.Conv2D(filters     = 32,
                  kernel_size = (3,3),
                  activation  = 'relu')(inputs)
x = layers.Conv2D(filters     = 64,
                  kernel_size = (3,3),
                  activation  = 'relu')(x)
block_1_output = layers.MaxPooling2D(pool_size = (3,3))(x)





# block 2
x = layers.Conv2D(filters     = 64, 
                  kernel_size = (3,3),
                  activation  = 'relu',
                  padding     = 'same')(block_1_output) # notice block_1_output
x = layers.Conv2D(filters     = 64, 
                  kernel_size = (3,3),
                  activation  = 'relu',
                  padding     = 'same')(x)
block_2_output = layers.add([x, block_1_output]) # notice block_1_output




# block 3
x = layers.Conv2D(filters     = 64, 
                  kernel_size = (3,3),
                  activation  = 'relu',
                  padding     = 'same')(block_2_output) # notice block_2_output
x = layers.Conv2D(filters     = 64, 
                  kernel_size = (3,3),
                  activation  = 'relu',
                  padding     = 'same')(x)
block_3_output = layers.add([x, block_2_output]) # notice block_2_output



# final output block
x = layers.Conv2D(filters     = 64,
                  kernel_size = (3,3),
                  activation  = 'relu')(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(units      = 256,
                 activation = 'relu')(x)
x = layers.Dropout(rate = 0.5)(x)
outputs = layers.Dense(units = 10)(x)




























# =============================================================================
# Create Model
# =============================================================================

# we use inputs and outputs from earlier
model = keras.Model(
    inputs  = inputs,  # inputs  from earlier
    outputs = outputs, # outputs from earlier
    name    = 'toy_resnet')

# let's check our toy ResNet model
model.summary()



























# =============================================================================
# Compile & Fit
# =============================================================================

# Compilation; determine appropriate 1) optimizer 2) loss 3) metrics
model.compile(
    optimizer = keras.optimizers.RMSprop(1e-3),
    loss      = keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics   = ["acc"]
    )


# Fit using our Preprocessed data
# I ONLY use first 5000 training images to keep training time short
model_fit = model.fit(
    x                = x_train_rescale[:5000],  # REscaled image data
    y                = y_train_OHE[:5000],      # One-Hot Encoded
    batch_size       = 64, 
    epochs           = 10, 
    validation_split = 0.1
    )

# let me visualize the training result next!























# =============================================================================
# Visualize Training result
# =============================================================================

# Training and Validation Accuracy records can be viewed like this:
model_fit.history


# Visualize Training & Validation Accuracies
plt.plot([i+1 for i in range(10)],
         model_fit.history['acc'],
         label = "Training Acc")

plt.plot([i+1 for i in range(10)],
         model_fit.history['val_acc'],
         label = "Validation Acc")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

'''
Using only the first 5000 training images,
we reached about 40% Training   Accuracy at the end of 10th Epoch,
       and about 45% Validation Accuracy


Remember there are 10 classes,
so a random guess would give us about 10% accuracy
'''

























"""
This is the end of "Non-linear Connectivity Topology" video~



Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""


















