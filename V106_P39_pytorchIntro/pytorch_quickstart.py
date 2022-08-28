#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 23:35:35 2022

@author: rj
"""
plt.subplot(221)
plt.imshow(test_data.data[1], cmap = "Greys")
plt.subplot(222)
plt.imshow(test_data.data[123], cmap = "Greys")
plt.subplot(223)
plt.imshow(test_data.data[3], cmap = "Greys")
plt.subplot(224)
plt.imshow(test_data.data[557], cmap = "Greys")









"""
Intro to PyTorch Neural Network
 (▰˘◡˘▰)
 


So far, all of my videos on Deep Learning / Neural Network used Tensorflow.
Hence, this is my 1st video using PyTorch.

PyTorch is a "very" popular Python package used for Deep Learning  🚀

A lot of explanation in this video is from:
    https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html


In this video,
A simple Neural Network will be trained to predict 10 labels/classes from the
famous FashionMNIST 👚👗 dataset. 
(4 FashionMNIST images on the right)


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""













# =============================================================================
# Import
# =============================================================================

import torch         # PyTorch
from torch import nn # needed for training neural network (nn)

# TorchVision contains FashionMNIST dataset
from torchvision import datasets

# DataLoader wraps an iterable around the Dataset - convenient!
from torch.utils.data import DataLoader

# following needed for converting data to tensors
from torchvision.transforms import ToTensor



# lastly, Numpy & Matplotlib
import numpy as np
import matplotlib.pyplot as plt



















# =============================================================================
# FashionMNIST 👚👗 Training & Test datasets
# =============================================================================

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



'''
If you are familiar with the MNIST Handwritten Digits dataset,
you may have noticed that MNIST & FashionMNIST have lots of similarities!🐣


If you want to read more about FashionMNIST dataset, 
please check out: https://github.com/zalandoresearch/fashion-mnist


According to the link:
    "We intend Fashion-MNIST to serve as a direct drop-in replacement for 
    the original MNIST dataset for benchmarking machine learning algorithms. 
    It shares the same image size and structure of training and testing 
    splits."
    (For viewers to read after pausing~ 👀)
'''














# =============================================================================
# DataLoader
# =============================================================================
'''
According to the Link I am referring to:
    "[DataLoader] wraps an iterable over our dataset, and supports automatic 
    batching, sampling, shuffling and multiprocess data loading. "  🚀
'''

# create DataLoader for Training data with batch size 64
train_dataloader = DataLoader(dataset    = train_data, 
                              batch_size = 64) 

# create DataLoader for Test data     with batch size 64
test_dataloader = DataLoader(dataset    = test_data, 
                             batch_size = 64)




# let's understand our DataLoader... 

# we can still see 60K Training samples like this
train_dataloader.dataset      # 60K datapoints
len(train_dataloader.dataset) # 60K

# 🚨 BUT the dataloader ITSELF is a bit DIFFERENT
len(train_dataloader) # 938 (as OPPOSED to 60K)

# remember we set our batch size to 64 (`batch_size = 64`) above
# if we multiply 64 by 938 
938 * 64 # 60032            <-- which is roughly 60K 🔥
# so DataLoader has divided 60K into 938 "groups"

# let's check 938 "groups"
for batch, (x, y) in enumerate(train_dataloader):
    print(batch + 1)    # +1 since it starts from 0
    print(f"shape of x: {x.shape}")
    print(f"shape of y: {y.shape} & dtype of y: {y.dtype}")
# notice the last batch is 938















# few more checks by me (not video) --------------
for batch, (x, y) in enumerate(train_dataloader):
    print(batch + 1)
    print(f"{x.shape}")   # [64, 1, 28, 28]
    print(f"{y}")
    break

for batch, (x, y) in enumerate(train_dataloader):
    print(batch + 1)
    print(f"{x}")
    print(f"{y}")
    break

for x, y in train_dataloader:
    print(f"{x.shape}")
    print(f"{y}")
    break

for x, y in train_data:
    print(f"{x.shape}")
    print(f"{y}")
    break
for i in test_data:
    print(i)














# =============================================================================
# Create Neural Network model 🚀
# =============================================================================
'''
According to the Link I am referring to:
    "To define a Neural Network in PyTorch, we create a class that inherits 
    from nn.Module." 
'''


# Define model
class NeuralNetwork(nn.Module): # inherit from nn.Module
    
    def __init__(self):
        '''
        define the layers of the network in the __init__ function
        '''
        super().__init__()     # super() to inherit methods from nn.Module
        
        # Flattening
        self.flatten = nn.Flatten()
        
        # Architecture
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features  = 28*28, # apply linear transformation
                      out_features = 512),
            nn.ReLU(),                      # apply rectified linear unit
            nn.Linear(in_features  = 512,   #       linear transformation
                      out_features = 512),
            nn.ReLU(),                      #       rectified linear unit
            nn.Linear(in_features  = 512,   #       linear transformation
                      out_features = 10)
            )
            
    def forward(self, x):
        '''
        specify how data pass through the network in the forward function
        using self.flatten() & self.linear_relu_stack() from __init__
        '''
        
        # flattening to get the correct in_features
        x = self.flatten(x)
        
        # Stacking Linear ReLU layers
        network = self.linear_relu_stack(x)
        return network


# check architecture
simpleNN = NeuralNetwork()
simpleNN   # notice flatten & linear_relu_stack created from above






















# del simpleNN



















# =============================================================================
# Loss Function & Optimizer
# =============================================================================

# Loss
loss_func = nn.CrossEntropyLoss()
# use CrossEntropy since this is 10-label classification problem

# Optimizer
opt = torch.optim.SGD(              # stochastic gradient descent
    params = simpleNN.parameters(), # simpleNN parameters to optimize
    lr     = 1e-3                   # learning rate 
    )

























# =============================================================================
# Training function    -  backpropagates prediction error
# =============================================================================

def training(dataloader, model, loss_func, optimizer):
    '''
    train simpleNN, while monitoring Loss
    '''
    
    size = len(dataloader.dataset) # 60K since there are 60K Training samples
    
    # Sets the module in training mode
    model.train()
    
    for batch, (x, y) in enumerate(dataloader): # training 1 batch at a time
        
        # Compute loss
        pred = model(x)           # simpleNN prediction 
        loss = loss_func(pred, y) # Loss calculation 
        
        
        # Backpropagation to adjust model's parameters
        optimizer.zero_grad() # zero out gradients in each loop
        loss.backward()       # computes gradient in reverse direction
        optimizer.step()      # conduct a single optimization step
        
        
        # Monitor for decrease in Loss as training progresses
        if batch % 100 == 0:
            loss    = loss.item()     # Loss
            current = batch * len(x)  # Progress
            print(f"loss: {loss:>5f} at {current:>5d}/{size:>5d}")


















# =============================================================================
# Test function   -  evaluate model against Test data
# =============================================================================
        
def testing(dataloader, model, loss_func):
    '''
    Evaluate simpleNN with Test data at the end of each epoch by
    monitoring the changes in Accuracy & Loss
    '''
    
    size = len(dataloader.dataset) # 10K, since there are 10K test samples
    num_batch = len(dataloader)    # 157 = len(test_dataloader)
    
    # sets the module in evaluation mode
    model.eval()
    
    # initialize monitoring values (starting from 0)
    test_loss, correct  = 0, 0
    
    with torch.no_grad(): # no_grad() to disable gradient calculation
        for x, y in dataloader:
            # prediction with simpleNN
            pred       = model(x) 
            
            # accumulate Loss and number of correct prediction
            test_loss += loss_func(pred, y).item()
            correct   += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    
    # updated monitoring value
    test_loss /= num_batch  # Average Loss
    correct   /= size       # Accuracy
    
    print(f"\n Test Evaluation: \n Accuracy: {(100*correct):>5f}%,\
    Avg loss: {test_loss:>5f} \n")





















# =============================================================================
# Train & Evaluation
# =============================================================================
'''
use training() & testing() functions we just defined earlier
'''


# number of epochs
epochs = 5

# train model & evaluate
for t in range(epochs):
    print(f"\n\n   Epoch {t+1}  -------------------- \n")
    
    # training
    training(train_dataloader, simpleNN, loss_func, opt)
    
    # testing
    testing(test_dataloader, simpleNN, loss_func)

'''
Notice Training Loss was greater than 2   during Epoch 1
                         smaller than 1.5 during Epoch 5
                         
Test Accuracy increases with each Epoch

They are good signs, but the final Test accuracy is only about 64%.
There are lots of room to improve! 😐
'''

















# =============================================================================
# Visualize & Predict 1st Test Image
# =============================================================================
'''
10 Labels from FashionMNIST can be found here: 
    https://github.com/zalandoresearch/fashion-mnist#labels
'''

# create labels using the same label names from the GitHub link
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


# very first Test image
plt.imshow(test_data.data[0], cmap = "Greys") # Ankle boot!
# pixel data
x = test_data[0][0] 
# true / correct label
y = test_data[0][1] 


# let's make prediction
with torch.no_grad():
    pred = simpleNN(x)                     # predict with simpleNN
    prediction = labels[pred[0].argmax(0)] # find predicted label
    truth      = labels[y]                 # true label
    
    print(f"simpleNN predict as {prediction} ; truth is {truth}")

# simpleNN correctly predicted: Ankle boot! 🙌



















"""
This is the end of "PyTorch Neural Network" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""


















