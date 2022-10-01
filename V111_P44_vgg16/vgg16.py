#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 21:39:40 2022

@author: rj
"""


from tensorflow.keras.backend import clear_session
clear_session()









"""
VGG16, Pretrained Network
 (▰˘◡˘▰)


When we have a 'small' dataset, we are UNlikely to train a model that can
generalize well. 

In such case, we use PRE-TRAINED network, which is a "saved network that was 
previously trained on a large dataset, typically on a large-scale 
image-classification task."


VGG16 is a Pretrained Convolutional Neural Network that is 16 layers deep 
and developed in 2014.
VGG16's architecture will be used to distinguish Dogs from Cats! 🐶😺



In previous video, I trained a ConvNet with Data Augmentation configuration,
and Validation Accuracy reached around 80%

Let's check if additionally using VGG16 Pretrained Network can BEAT that! 🚀 


-----


A lot of explanation in this video is from a book titled,
"Deep Learning with Python" (Chapter 5)


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""



















# =============================================================================
# 🐶😺 Image Dataset Directories
# =============================================================================
'''
This video classifies images as dogs 🐶 OR cats 😺.
Dogs & Cats image data is from Kaggle:
    https://www.kaggle.com/competitions/dogs-vs-cats/data

I already put image data into corresponding directories
'''

# import what I need
import os

# Directory where my data exist
mydata = "/Users/rj/Desktop/RJstudio/V110_P43_dataAugmentation/cat_dog_small"

# Directories to my Train, Validation, Test datasets
train_dir      = os.path.join(mydata, "train")
validation_dir = os.path.join(mydata, "validation")
test_dir       = os.path.join(mydata, "test")



# this way, my model knows where to find datasets later
# For example, my Training image is found in a folder 'cat_dog_small/train'















# =============================================================================
# Data Augmentation
# =============================================================================
'''
I am performing the SAME Random Image Transformation used in my previous
Data Augmentation YouTube video: https://youtu.be/GtaFu5Sevbk

(Also, more explanation about ImageDataGenerator in the previous video too)
'''

# import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Specify how to randomly transform / distort Training image data
train_dataAug_datagen = ImageDataGenerator(
    rescale            = 1./255,   # rescale
    rotation_range     = 40,       # randomly rotate  
    width_shift_range  = 0.2,      # randomly shift left or right
    height_shift_range = 0.2,      # randomly shift up or down
    zoom_range         = 0.3,      # randomly zooming in 
    horizontal_flip    = True,     # randomly flipping horizontally
    fill_mode          = 'nearest' # filling in newly created pixels
    )

# Validation image data "intentionally" NOT augmented / transformed 🚨
validation_datagen = ImageDataGenerator(
    rescale = 1./255) # ONLY rescale ; NO Data Augmentation



# Train Data Augmentation Generator
train_dataAug_generator = train_dataAug_datagen.flow_from_directory(
    directory   = train_dir,  # path to training 🐶 & 😺 images
    target_size = (150, 150), # resize images to 150 x 150
    batch_size  = 20,         # 20 images in each batch
    class_mode  = 'binary')   # since there are 2 labels: 🐶 & 😺
# Found 2000 images belonging to 2 classes:
    # 1000 🐶 & 1000 😺 images


# Validation Generator
validation_generator = validation_datagen.flow_from_directory(
    directory   = validation_dir,
    target_size = (150, 150),
    batch_size  = 20,
    class_mode  = 'binary')
# Found 1000 images belonging to 2 classes:
    # 500 🐶 & 500 😺 images

















# =============================================================================
# VGG16
# =============================================================================

# VGG16 comes prepackaged with Keras
from tensorflow.keras.applications import VGG16

# VGG16 🚀 
conv_base = VGG16(
    weights     = 'imagenet',   # VGG16 network trained on ImageNet
    include_top = False,        # 🚨 NOT include densely connected classifier 
    input_shape = (150, 150, 3) # input image tensors shape
    )
'''
Few things to know... 

"[VGG16 is] a simple and widely used convnet architecture for ImageNet"

ImageNet contains "1.4 million labeled images and 1,000 different classes"



Most IMPORTANTLY, notice I put `include_top = False`:
    This means I ONLY want to RE-USE Convolutional Base of VGG16, AND
    use my OWN densely connected classifier
    
    Representations learned by Convolutional Base are likely to be more 
    generic and therefore more reusable!
    
    BUT the representations learned by the classifier will necessarily be 
    specific to the set of classes on which the model was trained
    
    Therefore, I only RE-USE Convolutional Base and train my OWN classifier!
'''

# Check VGG16 Convolutional Base architecture
conv_base.summary()



# I do NOT want training process to alter what VGG16 already knows
# so I FREEZE 🧊 Convolutional Base so that weights are NOT updated
conv_base.trainable         # initially True
conv_base.trainable = False # set it to False
conv_base.trainable         # now False, FROZEN 🧊 cannot be updated




















# =============================================================================
# CNN Architecture with VGG16🌟
# =============================================================================

# import layers & models
from tensorflow.keras import layers, models


# build linear stack of layers sequentially, using `Sequential()`
vgg = models.Sequential()

# Use VGG16🌟 Convolutional Base, `conv_base` from earlier
vgg.add(conv_base)

# Flatten and then add my own CLASSIFIER
vgg.add(layers.Flatten())
vgg.add(layers.Dense(units = 128, activation = 'relu'))
vgg.add(layers.Dense(units = 1, activation = 'sigmoid')) # my own CLASSIFIER

# check full architecture
vgg.summary()

'''
Notice that VGG16 Convolutional Base has 
14 Million + parameters...

Ideally, one should be using GPU when the number of 
parameters is that large!
'''
























# =============================================================================
# Compile & Fit
# =============================================================================

# import optimizers
from tensorflow.keras import optimizers

# Compilation
vgg.compile(
    loss = 'binary_crossentropy',     # since this is "binary" problem
    optimizer = optimizers.RMSprop(),
    metrics = ['acc'])                # monitor accuracy during training / fit


# Fit
vgg_fit = vgg.fit(
    x                = train_dataAug_generator, # data provided by generator
    steps_per_epoch  = 100,
    epochs           = 5,
    validation_data  = validation_generator,    # data provided by generator
    validation_steps = 50
    )
# Since this Training process is computationally expensive,
# I only put 5 epochs! 🙏🏻


# View training result as Pandas DataFrame
import pandas as pd
pd.DataFrame(vgg_fit.history)


# Visualize Fit result
import matplotlib.pyplot as plt
plt.plot([i+1 for i in range(5)],
         vgg_fit.history['acc'],
         label = "Training Acc")
plt.plot([i+1 for i in range(5)],
         vgg_fit.history['val_acc'],
         label = "Validation Acc")
plt.legend(), plt.xlabel('Epochs'), plt.ylabel('Accuracy')

'''
Notice that with VGG16 Conv Base, Validation Accuracy is 
coming close to 90%

Without VGG16, CNN reached Validation Accuracy of ~80% 
at "50th" epoch.

So VGG16 Convolution Base did help my model reach higher 
Validation Accuracy.
'''




















"""
This is the end of "VGG16" video~

I wanted to share another important reason to Freeze 🧊 Convolutional Base
of Pre-trained network from the textbook:
    "
    Freezing a layer or set of layers means preventing their weights from being
    updated during training. 
    
    If you don’t do this, then the representations that were previously
    learned by the convolutional base will be modified during training. 
    
    Because the Dense layers on top are randomly initialized, very large 
    weight updates would be propagated through the network, effectively 
    destroying the representations previously learned.
    "


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""









