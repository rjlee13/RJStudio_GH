#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 00:20:57 2023

@author: rj
"""





import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 







"""
Character-Level Neural Language Model
 (▰˘◡˘▰)



Recently, Generative AI has brought massive excitement among AI community! 

I also regularly use ChatGPT on my smartphone, and Google released 
another incredible text Generative AI, called Bard. 



So I decided to make a video on Generative Deep Learning!
In this video, I train a very simple Character-Level Language model,
using only 1 LSTM layer + 1 Dense classifier layer  🚀



A lot of explanation in this video is from a book titled,
"Deep Learning with Python" (Chapter 8)


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""




















# =============================================================================
# Text Data: Nietzsche’s writing style 
# =============================================================================
'''
Need lots of Text Data
Use Nietzsche’s writings as our Text Data

So our language model will learn Nietzsche’s writing style:
https://s3.amazonaws.com/text-datasets/nietzsche.txt
'''

# import tensorflow & numpy
import tensorflow as tf
import numpy as np

# source of Nietzsche’s writing Text Data
path = tf.keras.utils.get_file(
    fname  = 'nietzsche.txt',
    origin = 'https://s3.amazonaws.com/text-datasets/nietzsche.txt'
    )

# take a look at our Nietzsche’s writing text data
text = open(path).read().lower() # convert to lowercase
text

# Character count 
print('Character count:', len(text)) # 600,893 characters(including spaces)

# Unique characters
chars = sorted(list(set(text)))
chars # take a look at unique characters
print('Number of unique characters:', len(chars)) # 57

# create a dictionary of unique characters
char_indices = dict((char, chars.index(char)) for char in chars)
char_indices
# for example letter 'x' has value (index) 50
char_indices['x']
















# =============================================================================
# Create Training Data
# =============================================================================

# want training sentences of 60 characters (including spaces)
maxlen = 60
# want new training sentences every 3 characters 
step = 3

# list of each training sentence 
sentences = []
# list of follow-up character of each training sentence 
next_chars = []


# populate lists: sentences & next_chars
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])   # populate sentences
    next_chars.append(text[i + maxlen])     # populate next_chars


# check first 5 training sentences
sentences[0:5] # notice consecutive sentences move up by 3 characters
# number of sentences collected
print('Number of sentences:', len(sentences))  # 200278
# so we have about 200K 60-character long training sentences

# check first 5 next_chars
next_chars[0:5]
# Same number of follow-up characters for each training sentence
print('Number of follow-up characters:', len(next_chars))  # 200278














# =============================================================================
# One-Hot Encoding
# =============================================================================
'''
Must "vectorize" our training text data before feeding it to Neural Network
'''

# initialize 2 arrays: x & y
    # array x collects One-Hot Encoded sentences
    # array y collects One-Hot Encoded follow-up character
x = np.zeros((len(sentences), maxlen, len(chars)), 
             dtype = np.bool)     
y = np.zeros((len(sentences), len(chars)), 
             dtype = np.bool)     
# `dtype = np.bool` means ALL array elements are initialized to False


# NOW change appropriate array elements to True (=1)
    # elements corresponding to each letter is changed to True from False
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]]   = 1 # changing to True
    y[i, char_indices[next_chars[i]]] = 1 # changing to True
    















# =============================================================================
# Neural Network Architecture + Compile + Train
# =============================================================================

# import what I need!
from tensorflow.keras import layers, models, optimizers

# Architecture: design a simple LSTM Network
lstmDense = models.Sequential()
lstmDense.add(layers.LSTM(                # single LSTM layer
    units       = 128,
    input_shape = (maxlen, len(chars))))
lstmDense.add(layers.Dense(               # Dense classifier
    units       = len(chars), 
    activation  = 'softmax'))


# check Architecture
lstmDense.summary() # ~ 102K parameters to train


# Compile: Assign appropriate optimizer & loss
lstmDense.compile(
    loss      = 'categorical_crossentropy', # multi-class classification
    optimizer = optimizers.RMSprop(learning_rate = 0.01))


# Fit: Start training for 5 epochs
lstmDense_fit = lstmDense.fit(
    x,
    y,
    batch_size = 256,
    epochs     = 5)

# Visualize how loss changes with each epoch 
import matplotlib.pyplot as plt
plt.plot([i+1 for i in range(5)],
         lstmDense_fit.history['loss'])
plt.ylabel('loss'); plt.xlabel('epoch')
# as we expect, loss is decreasing with each epoch














# =============================================================================
# Function to sample next / follow-up character
# =============================================================================
'''
this function helps us introduce "randomness" in our text generation
with the help of the second parameter, softmax temperature.

Low  Temperature => more deterministic; less random
High Temperature => more surprising   ; more random
'''

def sample(preds, temperature = 1.0):
    '''
    If you examine the function code, function is REweighting
    probability distribution coming out of model by temperature 
    
    Args:
        preds: model's predicted probability distribution
        (softmax) temperature: controls randomness 
    '''
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature  # REweighting by temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# =============================================================================
# Generating Text!
# =============================================================================

# import what I need
import random, sys

# randomly determine 60-character long start_sentence
start_index = random.randint(0, len(text)-maxlen-1)
start_sentence = text[start_index: start_index + maxlen]
print("--- Starting Sentence: "  + f"{start_sentence}")

# once start_sentence is fed to our lstmDense model,
# the model can start predicting follow-up characters! 🚀


# try 3 different softmax temperature values
# Remember:
    # Low  Temperature: more deterministic; less random
    # High Temperature: more surprising   ; more random
for temperature in [0.1, 0.5, 1.0]:
    print('\n\n ----- Temperature:', temperature, '\n')
    generated_text = start_sentence
    sys.stdout.write(generated_text)
    
    for i in range(200): # want model to predict follow-up 200 characters
        
        # One-Hot Encode 60-character long sentence
        sampled = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(generated_text):
            sampled[0, t, char_indices[char]] = 1.
        
        # predict a follow-up character after start_sentence
        preds = lstmDense.predict(sampled, verbose=0)[0]
        next_index = sample(preds, temperature) # <- sample() used here!
        next_char = chars[next_index]
        sys.stdout.write(next_char)
        
        # update text with predicted follow-up character
        generated_text += next_char
        generated_text = generated_text[1:]

'''
When softmax temperature is LOW (0.1),
the generated texts are quite repetitve; not interesting...


when softmax temeprature is HIGH (1.0),
the generated words are much more unpredictable/random. 😆
In fact, model has generated many strange NON-English words!!
'''
    

















"""
This is the end of "LSTM text generation" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""




















