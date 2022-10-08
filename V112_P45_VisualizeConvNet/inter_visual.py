#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 21:35:29 2022

@author: rj
"""


import numpy as np
from tensorflow.keras.backend import clear_session
clear_session()










"""
Visualize ConvNets
 (▰˘◡˘▰)


Convolutional Neural Networks (ConvNets) are widely used for Computer Vision
problems.
Ever wondered how ConvNets transform various input image data??!

This video visualizes how successive ConvNet layers are transforming 
input data images! 🚀


According to the textbook, I am reading:
    "The representations learned by ConvNets are highly amenable to 
    visualization, in large part because they’re representations of visual 
    concepts."

That sounds promising! 🙌


I re-use a ConvNet model trained in a previous video, and then visualize how
several convolution layers have transformed the input images.

I am showing how an intermediate Convolution layer decomposed a Cute Dog 👉

-----


A lot of explanation in this video is from a book titled,
"Deep Learning with Python" (Chapter 5)


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""





















# =============================================================================
# Data Directory
# =============================================================================

# import what I need
import os

# Directory where my data exist
mydata = "/Users/rj/Desktop/RJstudio/V110_P43_dataAugmentation/cat_dog_small"

# Directories to my Train, Validation, Test datasets
train_dir      = os.path.join(mydata, "train")
validation_dir = os.path.join(mydata, "validation")
test_dir       = os.path.join(mydata, "test")


# =============================================================================
# ImageDataGenerator
# =============================================================================

# import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Re-scale images by 1/255 so that pixel values fall in [0,1] interval
train_datagen = ImageDataGenerator(rescale = 1./255)
valid_datagen = ImageDataGenerator(rescale = 1./255)


# Train Generator
train_generator = train_datagen.flow_from_directory( # train_datagen from above
    directory   = train_dir,   # path to training 🐶 & 😺 images
    target_size = (150, 150),  # resize images to 150 x 150
    batch_size  = 20,          # 20 images in each batch
    class_mode  = 'binary')    # since there are 2 labels: 🐶 & 😺
# Found 2000 images belonging to 2 classes:
    # 1000 🐶 & 1000 😺


# Validation Generator
valid_generator = valid_datagen.flow_from_directory( # valid_datagen from above
    directory = validation_dir, # path to validation 🐶 & 😺 images
    target_size = (150, 150),   # resize images to 150 x 150
    batch_size  = 20,           # 20 images in each batch
    class_mode  = 'binary')     # since there are 2 labels: 🐶 & 😺
# Found 1000 images belonging to 2 classes:
    # 500 🐶 & 500 😺 images
    
    
    
    
    
    
    









    
# =============================================================================
# ConvNet Architecture (used in my previous YouTube video)
# =============================================================================
'''
This is the same ConvNet architecture used in my previous video 
where I introduced Data Augmentation ( https://youtu.be/GtaFu5Sevbk )

So detailed explanation about the architecture is 🚨SKIPPED🚨 in this video!
'''

# import layers and models
from tensorflow.keras import layers, models


# build linear stack of layers sequentially, using `Sequential()`
simple_cnn = models.Sequential()


# a stack of alternated Conv2D & MaxPooling2D layers
simple_cnn.add(layers.Conv2D(filters     = 32,
                             kernel_size = 3,
                             activation  = 'relu',
                             input_shape = (150, 150, 3)))
simple_cnn.add(layers.MaxPooling2D(pool_size = (2,2)))

simple_cnn.add(layers.Conv2D(filters     = 64,
                             kernel_size = 3,
                             activation  = 'relu'))
simple_cnn.add(layers.MaxPooling2D(pool_size = (2,2)))

simple_cnn.add(layers.Conv2D(filters     = 128,
                             kernel_size = 3,
                             activation  = 'relu'))
simple_cnn.add(layers.MaxPooling2D(pool_size = (2,2)))

simple_cnn.add(layers.Conv2D(filters     = 128,
                             kernel_size = 3,
                             activation  = 'relu'))
simple_cnn.add(layers.MaxPooling2D(pool_size = (2,2)))


# Flatten & Dense & Classifier output
simple_cnn.add(layers.Flatten())
simple_cnn.add(layers.Dense(units      = 512,
                            activation = 'relu'))
simple_cnn.add(layers.Dense(units      = 1,         # 1 unit => Classifier
                            activation = 'sigmoid'))


# check entire Architecture
simple_cnn.summary()

# Also...
simple_cnn.layers           # look up each layer like this
simple_cnn.layers[0]        # just look up the first layer like this
simple_cnn.layers[0].output # check output shape of the first layer like this



# =============================================================================
# Compile & Fit
# =============================================================================
'''
Compile & Fit to make sure my ConvNet model is learning
to distinguish Dogs from Cats! 🐶😺
'''

# import optimizers
from tensorflow.keras import optimizers

# Compile 
simple_cnn.compile(
    loss = 'binary_crossentropy',     # since this is "binary" problem
    optimizer = optimizers.RMSprop(),
    metrics = ['acc']) 

# Fit 
simple_cnn_fit = simple_cnn.fit(
    x                = train_generator,  # data provided by generator
    steps_per_epoch  = 100,
    epochs           = 30, 
    validation_data  = valid_generator,  # data provided by generator
    validation_steps = 50)


########### I already COMPILED & FIT by running the code above!


# Visualize Fit / Training result
import matplotlib.pyplot as plt
plt.plot([i+1 for i in range(30)],
         simple_cnn_fit.history['acc'],
         label = 'Training Acc')
plt.plot([i+1 for i in range(30)],
         simple_cnn_fit.history['val_acc'],
         label = 'Validation Acc')
plt.legend(), plt.xlabel("Epochs"), plt.ylabel("Accuracy")

'''
The ConvNet reaches almost perfect Training Accuracy at 30th epoch

            reaches about  70%  Validation  Accuracy at 30th epoch
'''













# =============================================================================
# A Cute Dog 🐶
# =============================================================================
'''
This Cute Dog's image will be used to find out how my ConvNet is
decomposing the image into several filters/channels in Convolution layers!
'''

# Path to the cute dog's image
cute_dog = "/Users/rj/Desktop/RJstudio/V110_P43_dataAugmentation/\
cat_dog_small/test/dog/dog.1759.jpg"

# import image
from tensorflow.keras.preprocessing import image


'''
Preprocess the image the SAME way other images were preprocessed for training
'''
# load image & resize to 150 x 150
dog = image.load_img(cute_dog, 
                     target_size = (150, 150))

# image to array 
dog_tensor = image.img_to_array(dog)
dog_tensor.shape # (150, 150, 3)

# reshape 
dog_tensor_reshape = np.expand_dims(dog_tensor, 
                                    axis = 0)
dog_tensor_reshape.shape # (1, 150, 150, 3)  <-- before it was (150, 150, 3)

# rescale
dog_tensor_reshape = dog_tensor_reshape / 255.0

# finally see our cute dog (after preprocessing)!
plt.imshow(dog_tensor_reshape[0]) # A Cute Dog 🐶


















# =============================================================================
# Keras Model with multiple outputs
# =============================================================================

# import models
from tensorflow.keras import models

# 'input' to Keras Model
simple_cnn.input

# 'output' to Keras Model
layer_output = [layer.output for layer in simple_cnn.layers[:8]]
layer_output
# All convolutional layers' outputs are stored in layer_output

# give input & output to Keras Model
visualizeModel = models.Model(inputs  = simple_cnn.input,
                              outputs = layer_output)


# now predict with our Cute Dog 🐶 `dog_tensor_reshape` from above
activations = visualizeModel.predict(dog_tensor_reshape)


# Cute Dog 🐶 after 1st layer
first_activation = activations[0]
plt.imshow(first_activation[0,:,:,5])   # 5th  channel
plt.imshow(first_activation[0,:,:,30])  # 30th channel

# Cute Dog 🐶 after 3rd layer
third_activation = activations[2]
plt.matshow(third_activation[0,:,:,5])  # 5th  channel
plt.matshow(third_activation[0,:,:,19]) # 19th channel
















# =============================================================================
# Many Channels together 🐕🦮
# =============================================================================

# 36 channels in the 1st layer
first_activation = activations[0]
for i in range(36):
    plt.subplot(6, 6, i+1)
    plt.imshow(first_activation[0, :, :, i],
               aspect = 'auto')
    plt.axis('off')
'''
Notice the outline of the Cute Dog 🐶 is
well preserved in most channel outputs
'''


# 36 channels in the 3rd layer
third_activation = activations[2]
for i in range(36):
    plt.subplot(6, 6, i+1)
    plt.imshow(third_activation[0, :, :, i],
               aspect = 'auto')
    plt.axis('off')
'''
Notice we can still see outline of the
Cute Dog 🐶
'''


# 36 channels in the 5th layer
fifth_activation = activations[4]
for i in range(36):
    plt.subplot(6, 6, i+1)
    plt.imshow(fifth_activation[0, :, :, i],
               aspect = 'auto')
    plt.axis('off')
'''
Notice that images look quite distored compared 
to the original Cute Dog 🐶

It is hard to see the 'cuteness'
'''


# 36 channels in the 7th layer
seventh_activation = activations[6]
for i in range(36):
    plt.subplot(6, 6, i+1)
    plt.imshow(seventh_activation[0, :, :, i],
               aspect = 'auto')
    plt.axis('off')
'''
We are NO longer able to see anything near our
Cute Dog 🐶
'''


'''
KEY 🔑 points to remember!!

1)  First few layers retain almost all the 
    information present in the initial 
    image
    
2)  After a few layers, we LOSE most of the
    visual contents of the original image,
    but the network gains more information 
    related to the CLASS of the image!
    
    (In this case, classes: Dog vs Cat)
'''


























"""
This is the end of "Visualize ConvNets" video~


I wanted to share the following passage from the textbook:
    
    "
    the features extracted by a layer become increasingly abstract with 
    the depth of the layer. 
    The activations of higher layers carry less and less information about 
    the specific input being seen, and more and more information about the 
    target
    "


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""












