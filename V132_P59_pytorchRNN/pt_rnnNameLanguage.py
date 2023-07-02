#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 15:27:33 2023

@author: rj

sources: 
https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
https://tutorials.pytorch.kr/intermediate/char_rnn_classification_tutorial.html
https://wikidocs.net/book/2788
"""









"""
Classifying Names using PyTorch RNN 
 (▰˘◡˘▰)

 
This video trains a Character-Level RNN model that predicts
which language a name is from based on the spelling, 
using PyTorch framework!



A lot of explanation is from:
https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""





















# =============================================================================
# Data Exploration (EDA)
# =============================================================================
'''
Download data from:
https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html#preparing-the-data
'''

# find ALL files with .txt extension
import glob
def findFiles(path):
    '''
    find files matching pattern described in path
    '''
    return glob.glob(path)
findFiles('./data/names/*.txt')
'''
['./data/names/Czech.txt', ...

there are 18 text files named as [Language].txt
so 18 languages
'''



# ASCII letters
import string
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters) # 57 letters

all_letters  # abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;'
n_letters    # 57 letters


# Turn a Unicode string to plain ASCII,
import unicodedata
def unicodeToAscii(s):
    '''
    copying answer from Stackoverflow: 
        https://stackoverflow.com/a/518232/2809427
    '''
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
unicodeToAscii('Ślusàrski') # 'Slusarski'  <- plain ASCII



# Try reading a file
from io import open
def readLines(filename):
    '''
    Read a file
    '''
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    #return lines
    return [unicodeToAscii(line) for line in lines]
readLines('./data/names/Korean.txt')
'''
['Ahn', 'Baik', 'Bang', 'Byon', 'Cha', 'Chang', ...

^ Korean.txt contains a bunch of Korean last names
'''



# Make a dict so that key = language,  value = last name of each language
import os 
category_lines = {} # dict mapping languages to a list of names 
all_categories = [] # all 18 languages in our data
for filename in findFiles('./data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0] # language
    all_categories.append(category)   # append language to all_categories
    lines = readLines(filename)
    category_lines[category] = lines

# check all_categories list
all_categories   # list of all 18 languages in our data
n_categories = len(all_categories)
n_categories     # 18 languages

# check category_lines dict 
category_lines['Korean'][:5]   # first 5 Korean names 
# ['Ahn', 'Baik', 'Bang', 'Byon', 'Cha']

category_lines['Japanese'][:5] # first 5 Japanese names 
# ['Abe', 'Abukara', 'Adachi', 'Aida', 'Aihara']











# =============================================================================
# Data Processing: Turning Names into Tensors
# =============================================================================

def letterToIndex(letter):
    '''
    each letter to index
    '''
    return all_letters.find(letter)
letterToIndex('a') # a -> 0
letterToIndex('c') # c -> 2



import torch
def letterToTensor(letter):
    '''
    each letter to tensor
    '''
    tensor = torch.zeros(1, n_letters) # recall n_letters = 57
    tensor[0][letterToIndex(letter)] = 1
    return tensor
letterToTensor('a') # we see 1 at 0th index position for a
letterToTensor('c') # we see 1 at 2nd index position for c



def lineToTensor(line):
    '''
    each word to tensor
    '''
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor
lineToTensor("ac") # 1 is positioned at places corresponding to a & c
lineToTensor("rj") # another example
# filled with 0s except for 1s at index of the current letter
# this is also called "one-hot vector"
















# =============================================================================
# RNN Architecture
# =============================================================================

import torch.nn as nn
class RNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        # 2 linear layers
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # LogSoftmax layer
        self.softmax = nn.LogSoftmax(dim=1) 

    def forward(self, input, hidden):
        '''
        let's check Architecture diagram: 
        https://tutorials.pytorch.kr/intermediate/char_rnn_classification_tutorial.html#id4 
        '''
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    
    def initHidden(self):
        '''
        initial hidden state
        '''
        return torch.zeros(1, self.hidden_size)

# try instantiating RNN class
n_hidden = 128
rnn = RNN(input_size  = n_letters,   # 57, number of letters
          hidden_size = n_hidden,    # 128
          output_size = n_categories # 18, number of languages
          ) 
# try
input  = lineToTensor('RJ')
hidden = torch.zeros(1, n_hidden)
output, next_hidden = rnn(input[0], hidden)
output
len(output[0]) # 18, each item is likelihood of 18 languages
# higher is more likely language














# =============================================================================
# Preparing for Training / Helper functions 
# =============================================================================

def categoryFromOutput(output):
    '''
    Helper Function:
        get the index of the greatest value (topk)
        tells you which one of 18 languages is mostly likely
    '''
    top_i = output.topk(1)[1]
    category_i = top_i[0].item()
    return all_categories[category_i], category_i
# example
categoryFromOutput(output) # picked Czech



import random
def randomChoice(l):
    '''
    pick random element from argument given
    '''
    return l[random.randint(0, len(l) - 1)]
randomChoice("ace") # randomly picking one of "a" "c" "e"



def randomTrainingExample():
    '''
    function for creating training samples
    '''
    category = randomChoice(all_categories) # randomly choose 1 of 18 languages
    category_tensor = torch.tensor(         # convert language to tensor
        [all_categories.index(category)],
        dtype = torch.long)
    line = randomChoice(category_lines[category]) # randomly choose name
    line_tensor = lineToTensor(line)              # convert name to tensor
    return category, line, category_tensor, line_tensor


# try randomTrainingExample()
for i in range(20): # randomly create 20 training samples
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)

















# =============================================================================
# Training Function
# =============================================================================

# NLLLOSS: The negative log likelihood loss. 
# often used to train a classification problem with C(=18) classes(=languages)
criterion = nn.NLLLoss()

# set learning rate
learning_rate = 0.005


def train(category_tensor, line_tensor):

    # initial hidden state
    hidden = rnn.initHidden()

    # Sets the gradients of all optimized torch.Tensors to 0
    rnn.zero_grad() 

    # RNN class used here; rnn
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    # loss calculation
    loss = criterion(output, category_tensor)
    loss.backward() # backward propagation

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha = -learning_rate)
    
    # return output and loss
    return output, loss.item()












# =============================================================================
# Conduct Training
# =============================================================================

import time
import math
def timeSince(since):
    '''
    function for calculating elapsed time in minute, second
    will be used to measure how much it takes to train
    '''
    now = time.time()
    s = now - since
    m = math.floor(s / 60) # minute
    s -= m * 60            # second
    return '%dm %ds' % (m, s)

# set variables
n_iters     = 100000
print_every = 5000
plot_every  = 1000

# Keep track of losses
current_loss = 0  # starting loss
all_losses   = [] # for collecting loss


start = time.time()
for iter in range(1, n_iters+1):
    
    # create training samples
    category, line, category_tensor, line_tensor = randomTrainingExample()
    
    # start training! 
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print ``iter`` number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % 
              (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss/plot_every)
        current_loss = 0
'''
Training Result:
    
    5000 5% (0m 1s) 0.7446 Marion / French ✓
    10000 10% (0m 3s) 0.7337 Shim / Korean ✓
    15000 15% (0m 5s) 2.5401 Almasi / Italian ✗ (Arabic)
    20000 20% (0m 7s) 1.7157 Mingo / Italian ✗ (Spanish)
    25000 25% (0m 9s) 0.2033 Pefanis / Greek ✓
    30000 30% (0m 10s) 0.7815 Faltejsek / Czech ✓
    35000 35% (0m 12s) 1.6600 Scott / English ✗ (Scottish)
    40000 40% (0m 14s) 0.9703 Houtem / Dutch ✓
    45000 45% (0m 16s) 1.4899 Oldham / Irish ✗ (English)
    50000 50% (0m 17s) 0.5592 Ferreiro / Portuguese ✓
    55000 55% (0m 19s) 5.0367 Castillion / Russian ✗ (Spanish)
    60000 60% (0m 21s) 3.1668 Hayden / Dutch ✗ (Irish)
    65000 65% (0m 23s) 1.0696 Bueren / Dutch ✓
    70000 70% (0m 25s) 6.5887 Cernohous / Irish ✗ (Czech)
    75000 75% (0m 26s) 0.1726 Kowalski / Polish ✓
    80000 80% (0m 28s) 2.9254 Newstead / Chinese ✗ (English)
    85000 85% (0m 30s) 1.4956 Kieu / Chinese ✗ (Vietnamese)
    90000 90% (0m 32s) 0.6082 Youj / Korean ✓
    95000 95% (0m 34s) 0.1389 Que / Chinese ✓
    100000 100% (0m 35s) 1.0437 Fukao / Japanese ✓

'''














# =============================================================================
# Prediction
# =============================================================================

def evaluate(line_tensor):
    '''
    the same as train() function minus the backprop
    '''

    # zeroed initial hidden state
    hidden = rnn.initHidden()

    # RNN class used here; rnn
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    # just return output
    return output



def predict(input_line, n_prediction = 3):
    '''
    predict 3 likely languages a name is from based on spelling
    '''
    print(f'\n {input_line}')

    with torch.no_grad(): # disable gradient calculation
        
        # get output after converting input_line to tensor
        output = evaluate(lineToTensor(input_line))

        # get 3 top predictions
        topv, topi = output.topk(k       = n_prediction, 
                                 dim     = 1, 
                                 largest = True)

        # print predictions nicely
        for i in range(n_prediction):
            value = topv[0][i].item()          # likelihood
            category_index = topi[0][i].item() # predicted language
            print('(%.2f) %s' % (value, all_categories[category_index]))


# try predicting!

predict('Kim')  # predicting Korean most likely 

predict('Son')

predict('Shinji')

predict('James')
















"""
This is the end of "PyTorch RNN" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""






















