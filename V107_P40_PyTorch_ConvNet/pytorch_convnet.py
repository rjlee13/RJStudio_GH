#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 00:21:22 2022

@author: rj
"""

https://medium.com/analytics-vidhya/pytorch-for-deep-learning-convolutional-
neural-networks-fashion-mnist-f7bff7b4e724

https://www.kaggle.com/code/pankajj/fashion-mnist-with-pytorch-93-accuracy/
notebook

import numpy as np
import matplotlib.pyplot as plt















"""
PyTorch Convolutional Neural Networks (ConvNet)
 (▰˘◡˘▰)

 

This video trains a PyTorch ConvNet to classify 10 different clothings
from the famous Fashion-MNIST 👚👗 dataset.

In a previous video, a VERY simple 3-layer model (NOT ConvNet) achieved 
~64% accuracy on Test dataset.
(I also recommend you to watch that video if you are new to PyTorch! 😄)

We will see that ConvNet can boost Test dataset Accuracy by LOTS 🚀🚀




I SKIP detailed explanations on ConvNet. However, I created a video to 
cover lots of basic concepts of ConvNet in another video some time ago!
If you are interested, please check it out!

Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""



















# =============================================================================
# FashionMNIST 👚👗
# =============================================================================

# TorchVision contains FashionMNIST dataset
from torchvision import datasets
# following is needed for converting data to tensors
from torchvision.transforms import ToTensor


# download Training data
train_data = datasets.FashionMNIST(
    root      = "data",
    train     = True,       # for Training data
    download  = True,
    transform = ToTensor()  # for converting to tensor
    )

# quick check
train_data            # 60K training samples
train_data.data.shape # each sample is 28x28 grayscale image
np.unique(train_data.targets, return_counts = True)
# 10 unique labels: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# each label appearing 6K times



# download Test data
test_data = datasets.FashionMNIST(
    root      = "data",
    train     = False,      # for Test data
    download  = True,
    transform = ToTensor()  # for converting to tensor
    )


# quick check
test_data             # 10K test samples
test_data.data.shape  # each sample is 28x28 grayscale image
np.unique(test_data.targets, return_counts = True)
# 10 unique labels: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# each label appearing 1K times
















# =============================================================================
# DataLoader
# =============================================================================

# import DataLoader that wraps an iterable around the Dataset
from torch.utils.data import DataLoader


# create DataLoader for Training data with batch size 150
train_Dloader = DataLoader(dataset    = train_data, 
                           batch_size = 150) 

# create DataLoader for Test data     with batch size 150
test_Dloader =  DataLoader(dataset    = test_data, 
                           batch_size = 150)




# let's do a quick check with one of the Data Loaders


# total number of Test dataset
len(test_Dloader.dataset)     # 10000
# Length of Test Data Loader
len(test_Dloader)             # 67 batches

# let's print out 67 batches
for index, (image, label) in enumerate(test_Dloader):
    print(index + 1) 
    print(image.shape)
    print(label.shape)
# batches have size 150 as we set above (`batch_size = 150`)
# except the last one (100)

# So...
66 * 150 + 100  # 10000     <-- Sample size of Test data






##### some more personal check
for index, (image, label) in enumerate(test_Dloader):
    print(index + 1) 
    print(len(image))
    print(label.shape)











# =============================================================================
# Visualization ✨
# =============================================================================

# Labels from Fashion-MNIST GitHub page:
    # https://github.com/zalandoresearch/fashion-mnist#labels
labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]



# Simple function to visualize and find label of a training sample! 
def fashionShow(index):
    plt.imshow(train_data.data[index], cmap = "Greys")
    print(labels[train_data[index][1]])

# For example, 3rd index training sample is a Dress 👗
fashionShow(3)

# 89th index training sample is an Ankle boot 🥾
fashionShow(89) 









##### some more personal check
train_data[0][0]    
train_data[0][0].shape # [1, 28, 28]  <- this is actual pixel data


train_data[0][1]       # 9            <- this is label














# =============================================================================
# Define Convolutional Neural Network 🎖️
# =============================================================================

# import PyTorch
import torch         
from torch import nn


# Convnet Class
class fashionConvNet(nn.Module): # inherits from nn.Module

    '''
    define the layers of the network in the __init__ function
    '''
    def __init__(self):
        super().__init__()
        
        
        # 1st convolution layer 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels       = 1,
                      out_channels      = 32,
                      kernel_size       = 2,
                      padding           = 'same'),
            nn.BatchNorm2d(num_features = 32),       # batch norm
            nn.ReLU(),                               # ReLU activation
            nn.MaxPool2d(kernel_size    = 2)         # MaxPooling
            )
        
        
        # 2nd convolution layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels       = 32,
                      out_channels      = 64,
                      kernel_size       = 5),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size    = 2)
            )
        
        
        # Flattening for the upcoming Dense layers
        self.flatten = nn.Flatten()
        
        
        # dense layer with dropout
        self.fc1 = nn.Linear(
            in_features  = 64*5*5,
            out_features = 512)
        self.drop1 = nn.Dropout2d(p = 0.3)     # dropout
        
        
        # dense layer with dropout once more
        self.fc2 = nn.Linear(
            in_features  = 512,
            out_features = 128)
        self.drop2 = nn.Dropout2d(p = 0.2)     # dropout
        
        
        # batch normalization & output layer
        self.batchNorm = nn.BatchNorm1d(num_features = 128)
        self.fc3 = nn.Linear(
            in_features = 128,
            out_features = 10)

    '''
    specify how data pass through the network in forward function
    '''
    def forward(self, x):
        out = self.conv1(x)       # conv layer
        out = self.conv2(out)     # conv layer
        out = self.flatten(out)   # flattening

        out = self.fc1(out)       # dense layers + dropout + batch norm 
        out = self.drop1(out)     
        out = self.fc2(out)
        out = self.drop2(out)
        out = self.batchNorm(out)
        out = self.fc3(out)       # final output
        
        return out
        
    


convModel = fashionConvNet()

# check architecture
print(convModel)
























'''
I tried the following to find out the output shape of the convolution layers:
    
    

foo = nn.Sequential(
    nn.Conv2d(in_channels = 1,
              out_channels = 32,
              kernel_size = 2,
              padding = 'same'),
    nn.BatchNorm2d(num_features = 32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size = 2)
    )

foo(torch.randn(1,1,28,28)).shape


poo = nn.Sequential(
    nn.Conv2d(in_channels = 32,
              out_channels = 64,
              kernel_size = 5),
    nn.BatchNorm2d(num_features = 64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size = 2)
    )

poo(torch.randn(1,32,14,14)).shape
'''



















# =============================================================================
# Loss & Optimizer
# =============================================================================

# Loss Function 
loss_fn = nn.CrossEntropyLoss()
# using CrossEntropy since this is 10-label classification problem

# Optimizer 
optimizer = torch.optim.Adam(  # implement Adam algorithm
    convModel.parameters(),    # convModel parameters to optimize
    lr = 0.002                 # learning rate
    )























# =============================================================================
# Training Function
# =============================================================================

def training_convnet(dataloader, model, loss_fn, optimizer):
    '''
    function for training convModel with train_Dloader
    '''
    
    size = len(dataloader.dataset) # 60K, since we have 60K Train images
    
    # Sets the module in training mode
    model.train()
    
    for index, (image, label) in enumerate(dataloader):
        
        
        # Loss calculation
        pred = model(image)         # convModel prediction
        loss = loss_fn(pred, label) # Loss calculation
        
        
        # Backpropagation
        optimizer.zero_grad() # zero out gradients in each loop
        loss.backward()       # compute gradient
        optimizer.step()      # conduct optimization step
        
        
        # print loss and progress of training
        if index % 100 == 0:
            loss = loss.item()
            current = index * len(image)
            print(f"loss: {loss:.3f}\
            progress: {(current / size)*100:.2f}%")














# check if it works
# training_convnet(train_Dloader, convModel, loss_fn, optimizer)












# =============================================================================
# Testing Function
# =============================================================================

def testing_convnet(dataloader, model):
    '''
    function for evaluating convModel with test_Dloader
    '''
    
    size = len(dataloader.dataset) # 10K, since we have 10K Test images
    
    # sets the module in evaluation mode
    model.eval()
    
    # initialize correct prediction count (starting from 0)
    correct = 0
    
    with torch.no_grad(): # no_grad() to disable gradient calculation
        for image, label in dataloader:
            
            # prediction with convModel
            pred = model(image)
            
            # update number of correct prediction values
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
            
    print(f"\n  Test Accuracy: {(correct/size)*100:.2f}%")
















# check if it works
# testing_convnet(test_Dloader, convModel)
















# =============================================================================
# Train & Evaluation
# =============================================================================
'''
use training_convnet() & testing_convnet() functions defined earlier
'''


epochs = 5

for e in range(epochs):
    
    # print Epoch number
    print(f"\n\n   Epoch {e+1}  -------------------- \n")
    
    # training
    training_convnet(train_Dloader, convModel, loss_fn, optimizer)
    
    # evaluating at the end of each epoch
    testing_convnet(test_Dloader, convModel)





'''
In the beginning, I mentioned a video training VERY simple Neural Network 
that achieved ~64% Test Accuracy.


With 2 Convolutional Layers + 3 fully connected layer, 
we achieved ~90% Test Accuracy 🎉 
More than 20% increase!


We can see Test Accuracy achieved by many other smart people too!
    from https://github.com/zalandoresearch/fashion-mnist#benchmark
    
'''


























"""
This is the end of "PyTorch Neural Network" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""


















