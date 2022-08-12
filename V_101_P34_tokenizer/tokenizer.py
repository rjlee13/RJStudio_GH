#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 12:15:53 2022

@author: rj
"""


















"""
Tokenizer
 (▰˘◡˘▰)



When working with TEXT data, you are very likely to encounter
the concept called "Tokenization". 

In short, the process of splitting text into tokens (words, characters, etc)
is called Tokenization. 🚀

Lots of text-vectorization (used lots in Neural Network) applies some type of
Tokenization to map tokens to vectors!

In this short video, I show how to Tokenize using Tensorflow Keras 🔥



A lot of explanation in this video is from a book titled,
"Deep Learning with Python" (Chapter 6)


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""

# =============================================================================
# Import
# =============================================================================

# this is ALL I use / need for this short video
from tensorflow.keras.preprocessing.text import Tokenizer
















# =============================================================================
# Tokenizer()
# =============================================================================

# sample sentences
sentences = ['Today is Friday', 'Today is hot', 'Greece is cool'] # 3 sentences

# Tokenizer() 
tokenizer = Tokenizer(num_words = 8)
# `num_words = 8` means I am going to keep the MOST COMMON 7 (= 8-1) words

# create an internal vocabulary list / create word index
# use fit_on_texts()
tokenizer.fit_on_texts(sentences)



'''
1 Convert strings to integers, using texts_to_sequences()
'''
sequences = tokenizer.texts_to_sequences(sentences)
sequences
# so 'Today'  is 2
#    'is'     is 1
#    'Friday' is 3
#    'Greece' is 5

# NOTICE that 'is' is 1 ✨ That is because 'is' is the MOST COMMON
# 'is' appears 3 times

# 'Today' is 2 because it is the SECOND MOST common word
# 'Today' is appearing 2 times

'''
2 View word index created, using word_index
'''
word_index = tokenizer.word_index
word_index         
word_index.items() # just another way to view word index
print('Found %s unique tokens.' % len(word_index)) # 6 unique tokens

# simple for loop to see word index:
for word, index in word_index.items():
    print(index, word)
# Indeed, we can check again that 
#    'is'     is 1
#    'Greece' is 5

# to get integer index for a particular word, use get('word')
word_index.get('is')    # 1, since it is MOST COMMON word
word_index.get('today') # 2, since it is SECOND MOST COMMON word


'''
3 One Hot Encoding, using texts_to_matrix(<sentences>, mode = 'binary')
'''
OH_encode = tokenizer.texts_to_matrix(sentences, mode = 'binary')
OH_encode
# [0., 1., 1., 1., 0., 0., 0., 0.] -> Today is Friday
# [0., 1., 1., 0., 1., 0., 0., 0.] -> Today is hot
# [0., 1., 0., 0., 0., 1., 1., 0.] -> Greece is cool

# Again note that 
# this ONE (see mouse cursor) is 'is'      <- MOST COMMON word
# this ONE (see mouse cursor) is 'today'   <- SECOND MOST COMMON
# etc


















"""
This is the end of "Tokenizer" video~


Hope you enjoyed and learned something new!
Thank you for watching ◎[▪‿▪]◎ 
"""



















