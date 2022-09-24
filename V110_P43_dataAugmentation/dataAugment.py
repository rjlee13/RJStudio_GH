#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 15:44:20 2022

@author: rj
"""
from tensorflow.keras.backend import clear_session
clear_session()











"""
Data Augmentation: Dogs vs. Cats classification
 (▰˘◡˘▰)


This video classifies images as dogs 🐶 OR cats 😺.
Dogs & Cats image data is from Kaggle:
    https://www.kaggle.com/competitions/dogs-vs-cats/data


Because I ONLY use 2000 images as training data, 
our Neural Network (namely, CNN) starts to OVERFIT very quickly 🚨


This is where "Data Augmentation" technique kicks in to mitigate 
overfitting in computer vision problems 🚀 
GOAL of this video is to introduce Data Augmentation using Tensorflow.
I have an example of "Data-Augmented" Cat image on the right 👉


-----


To focus on Data Augmentation, I SKIP steps to download and divide data
into Training, Validation, and Test sets
BUT if you want to see the entire code, it is available from my GitHub:
    https://github.com/rjlee13/RJStudio_GH


-----


A lot of explanation in this video is from a book titled,
"Deep Learning with Python" (Chapter 5)


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""



















# =============================================================================
# Kaggle Dogs vs Cats 
# =============================================================================

# original Kaggle data
kaggle_data = "/Users/rj/Desktop/RJstudio/V110/dogs-vs-cats/train"

# Directory to put my data
mydata = "/Users/rj/Desktop/RJstudio/V110/cat_dog_small"





# =============================================================================
# Create train, validation, test directories
# =============================================================================

# import what I need
import os, shutil


# train directory
train_dir = os.path.join(mydata, "train")
train_dir # '/Users/rj/Desktop/RJstudio/V110/cat_dog_small/train'
os.mkdir(train_dir)

# validation directory
validation_dir = os.path.join(mydata, "validation")
os.mkdir(validation_dir)

# test directory
test_dir = os.path.join(mydata, "test")
os.mkdir(test_dir)





# =============================================================================
# Create Cat & Dog directories
# =============================================================================

# train cat & dog
train_cat_dir = os.path.join(train_dir, 'cat')
os.mkdir(train_cat_dir)

train_dog_dir = os.path.join(train_dir, 'dog')
os.mkdir(train_dog_dir)


# validation cat & dog
validation_cat_dir = os.path.join(validation_dir, 'cat')
os.mkdir(validation_cat_dir)

validation_dog_dir = os.path.join(validation_dir, 'dog')
os.mkdir(validation_dog_dir)


# test cat & dog
test_cat_dir = os.path.join(test_dir, 'cat')
os.mkdir(test_cat_dir)

test_dog_dir = os.path.join(test_dir, 'dog')
os.mkdir(test_dog_dir)







# =============================================================================
# Copy Images to my directories
# =============================================================================

# Training Cat 😺
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
fnames

for fname in fnames:
    src = os.path.join(kaggle_data, fname)
    dst = os.path.join(train_cat_dir, fname)
    shutil.copyfile(src, dst)


# Validation Cat 😺
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
fnames

for fname in fnames:
    src = os.path.join(kaggle_data, fname)
    dst = os.path.join(validation_cat_dir, fname)
    shutil.copyfile(src, dst)


# Test Cat 😺
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
fnames

for fname in fnames:
    src = os.path.join(kaggle_data, fname)
    dst = os.path.join(test_cat_dir, fname)
    shutil.copyfile(src, dst)




# Training Dog 🐶
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
fnames

for fname in fnames:
    src = os.path.join(kaggle_data, fname)
    dst = os.path.join(train_dog_dir, fname)
    shutil.copyfile(src, dst)


# Validation Dog 🐶
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
fnames

for fname in fnames:
    src = os.path.join(kaggle_data, fname)
    dst = os.path.join(validation_dog_dir, fname)
    shutil.copyfile(src, dst)


# Test Dog 🐶
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
fnames

for fname in fnames:
    src = os.path.join(kaggle_data, fname)
    dst = os.path.join(test_dog_dir, fname)
    shutil.copyfile(src, dst)




# =============================================================================
# Sanity Check - check number of images 
# =============================================================================

# Training Cat & Dog Images
print(f"Number Training Cat Images {len(os.listdir(train_cat_dir))}") # 1000
print(f"Number Training Dog Images {len(os.listdir(train_dog_dir))}") # 1000
# 1000 training images of 🐶 & 😺 each



# Validation Cat & Dog Images
print(f"Number Val Cat Images {len(os.listdir(validation_cat_dir))}") # 500
print(f"Number Val Dog Images {len(os.listdir(validation_dog_dir))}") # 500
# 500 validation images of 🐶 & 😺 each



# Test Cat & Dog Images 
print(f"Number Test Cat Images {len(os.listdir(test_cat_dir))}") # 500
print(f"Number Test Dog Images {len(os.listdir(test_dog_dir))}") # 500
# 500 test images of 🐶 & 😺 each















# =============================================================================
# ImageDataGenerator (NO Data Augmentation)
# =============================================================================
'''
To demonstrate effectiveness of Data Augmentation,
my first model does NOT use Data Augmentation configuration 🚨
'''

# import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
'''
According to the textbook:
    "[ImageDataGenerator] lets you quickly set up Python generators that can 
    automatically turn image files on disk into batches of preprocessed 
    tensors"
'''


# Re-scale images by 1/255 so that pixel values fall in [0,1] interval
train_datagen = ImageDataGenerator(rescale = 1./255)
valid_datagen = ImageDataGenerator(rescale = 1./255)
'''
ONLY Re-scaling and NO Data Augmentation performed by ImageDataGenerator
for now.

Just remember ImageDataGenerator specifies how to perform Data Augmentation, 
which will be shown later in the video.
'''


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






# Let's take a look at what Validation Generator can do for us
for data_batch, label_batch in valid_generator:
    print(f'data_batch shape : {data_batch.shape}')
    print(f'label_batch shape: {label_batch.shape}')
    break # break to just check out the 1st batch

# 20 images in this batch          (batch_size == 20)
# images are resized to 150 x 150  (target_size) 
# 3 channels: Red Green Blue (RGB) 
    
















# =============================================================================
# Simple CNN's Architecture (NO Data Augmentation)
# =============================================================================
'''
Classic CNN architecture is used: 
    Few Conv2D & Maxpooling layers and then
    Flatten -> Dense -> Classifier (Dense with sigmoid + 1 unit)
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


# Flatten & Dense 
simple_cnn.add(layers.Flatten())
simple_cnn.add(layers.Dense(units      = 512,
                            activation = 'relu'))
simple_cnn.add(layers.Dense(units      = 1,
                            activation = 'sigmoid'))


# check architecture
simple_cnn.summary()

















# =============================================================================
# Compile & Fit (NO Data Augmentation)
# =============================================================================

# import optimizers
from tensorflow.keras import optimizers

# Compile
simple_cnn.compile(
    loss = 'binary_crossentropy',     # since this is "binary" problem
    optimizer = optimizers.RMSprop(),
    metrics = ['acc'])                # monitor accuracy during training / fit


'''
Fit ... Few things to know here

1) Train & Validation data are provided by generators created earlier:
    train_generator & valid_generator
2) steps_per_epoch is 100 because the batch size is 20, and 20*100 = 2000,
   which is the total number of Training images.
3) validation_steps is 50 because the batch size is 20, and 20*50 = 1000,
   which is the total number of Validation images.
'''
simple_cnn_fit = simple_cnn.fit(
    x                = train_generator,  # data provided by generator
    steps_per_epoch  = 100,
    epochs           = 50, 
    validation_data  = valid_generator,  # data provided by generator
    validation_steps = 50)


# Fit result as Pandas DataFrame
import pandas as pd
pd.DataFrame(data = simple_cnn_fit.history)


# Visualize Fit result
import matplotlib.pyplot as plt
plt.plot([i+1 for i in range(50)],
         simple_cnn_fit.history['acc'],
         label = 'Training Acc')
plt.plot([i+1 for i in range(50)],
         simple_cnn_fit.history['val_acc'],
         label = 'Validation Acc')
plt.legend(), plt.xlabel("Epochs"), plt.ylabel("Accuracy")

'''
Notice how Validation Accuracy stays around ~70%,
and it does NOT look like it's going to improve anymore...


Let's see if Data Augmentation can help!! 🌟
'''





















# =============================================================================
# Visualize Data Augmentation 
# =============================================================================

# Augment data by random transformations 🤖 using ImageDataGenerator
toy_datagen = ImageDataGenerator(
    rotation_range     = 30,       # randomly rotate  
    width_shift_range  = 0.2,      # randomly shift left or right
    height_shift_range = 0.2,      # randomly shift up or down
    zoom_range         = 0.3,      # randomly zooming in 
    horizontal_flip    = True,     # randomly flipping horizontally
    fill_mode          = 'nearest' # filling in newly created pixels
    )
# Now let me visualize few examples of above random transformations


# import image
from tensorflow.keras.preprocessing import image

# choose one image and see withOUT transformation
one_image = '/Users/rj/Desktop/RJstudio/V110/cat_dog_small/train/cat/cat.977.jpg'
img = image.load_img(one_image,               # read image
                     target_size = (150,150)) # resize to 150x150
img # it's a 😺


# Convert the 😺 image to a Numpy array with shape (150, 150, 3)
x = image.img_to_array(img)
x.shape # (150, 150, 3)


# Reshape 
x = x.reshape((1,) + x.shape)
x.shape # (1, 150, 150, 3)   <-- before .reshape(), it was (150, 150, 3)


# Visualize 4 transformations
i = 0
# use `toy_datagen` from above to perform Data Augmentation
for batch in toy_datagen.flow(x = x, batch_size = 1):
    plt.subplot(2, 2, i+1)  # put 4 transformations together
    plt.imshow(image.array_to_img(batch[0])) # visualize
    i += 1
    if i % 4 == 0:
        break
'''
4 randomly transformed / distorted images of the 😺.

As you can see, Data Augmentation generates "believable-looking" images 
to expose our model to more aspects of the data and generalize better!!
'''



















# =============================================================================
# Apply Data Augmentation to our 🐶 & 😺 image data
# =============================================================================

# Specify how to randomly transform training image data
train_dataAug_datagen = ImageDataGenerator(
    rescale            = 1./255,   # rescale
    rotation_range     = 40,       # randomly rotate  
    width_shift_range  = 0.2,      # randomly shift left or right
    height_shift_range = 0.2,      # randomly shift up or down
    zoom_range         = 0.3,      # randomly zooming in 
    horizontal_flip    = True,     # randomly flipping horizontally
    fill_mode          = 'nearest' # filling in newly created pixels
    )

# Validation image data "intentionally" NOT augmented 🚨
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
# CNN architecture with Data Augmention
# =============================================================================

# build linear stack of layers sequentially, using `Sequential()`
dataAug_cnn = models.Sequential()

# a stack of alternated Conv2D & MaxPooling2D layers
dataAug_cnn.add(layers.Conv2D(filters     = 32,
                              kernel_size = 3,
                              activation  = 'relu',
                              input_shape = (150, 150, 3)))
dataAug_cnn.add(layers.MaxPooling2D(pool_size = (2,2)))

dataAug_cnn.add(layers.Conv2D(filters     = 64,
                              kernel_size = 3,
                              activation  = 'relu'))
dataAug_cnn.add(layers.MaxPooling2D(pool_size = (2,2)))

dataAug_cnn.add(layers.Conv2D(filters     = 128,
                              kernel_size = 3,
                              activation  = 'relu'))
dataAug_cnn.add(layers.MaxPooling2D(pool_size = (2,2)))

dataAug_cnn.add(layers.Conv2D(filters     = 128,
                              kernel_size = 3,
                              activation  = 'relu'))
dataAug_cnn.add(layers.MaxPooling2D(pool_size = (2,2)))


# Flatten, Dropout, Dense 
dataAug_cnn.add(layers.Flatten())
dataAug_cnn.add(layers.Dropout(rate = 0.3))  
dataAug_cnn.add(layers.Dense(units = 512,
                             activation = 'relu'))
dataAug_cnn.add(layers.Dense(units = 1,
                             activation = 'sigmoid'))

# Check architecture
dataAug_cnn.summary()
















# =============================================================================
# Compile & Fit
# =============================================================================

# Compilation
dataAug_cnn.compile(
    optimizer = optimizers.RMSprop(),
    loss = 'binary_crossentropy',
    metrics = ['acc'])


# Fit
dataAug_cnn_fit = dataAug_cnn.fit(
    x                = train_dataAug_generator, # data provided by generator
    steps_per_epoch  = 100,
    epochs           = 50,
    validation_data  = validation_generator,    # data provided by generator
    validation_steps = 50
    )




# Visualize Fit result
plt.plot([i+1 for i in range(50)],
         dataAug_cnn_fit.history['acc'],
         label = 'Training Acc')
plt.plot([i+1 for i in range(50)],
         dataAug_cnn_fit.history['val_acc'],
         label = 'Validation Acc')
plt.legend(), plt.xlabel("Epochs"), plt.ylabel("Accuracy")
'''
After applying Data Augmentation, we can see that 
both Training and Validation Accuracies showing increasing trend
even at around 50th epoch.

we did NOT see this when Data Augmentation was NOT used! 
'''








# Visualize Data Augmentation effect
plt.plot([i+1 for i in range(50)],
         simple_cnn_fit.history['val_acc'],
         label = 'NO Data Augmentation')
plt.plot([i+1 for i in range(50)],
         dataAug_cnn_fit.history['val_acc'],
         label = 'Data Augmentation')
plt.legend(), plt.xlabel("Epochs"), plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
'''
WithOUT Data Augmentation, Validation Accuracy stalls at around 70%

With    Data Augmentation, Validation Accuracy shows increasing trend! 🎉
'''





















"""
This is the end of "Data Augmentation" video~


I copied the following passage from the textbook:
    
    "If you train a new network using this data-augmentation configuration, 
    the network will never see the same input twice. 
    
    But the inputs it sees are still heavily intercorrelated, because they 
    come from a small number of original images—you can’t produce new 
    information, you can only remix existing information. As such, this may 
    not be enough to completely get rid of overfitting."
    
    (Thought 👆 was important)



Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""











