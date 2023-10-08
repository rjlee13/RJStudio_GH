#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 17:22:48 2023

@author: rj
"""











"""
Intro to Hugging Face: Pretrained Model, DistilBERT
 (▰˘◡˘▰)


Hugging Face (HF) is a popular AI hub that provides tools for us to use!
Link ->  https://huggingface.co/



This video uses HF DistilBERT, a smaller + faster distilled version of BERT,
to conduct classification task on Movie Review dataset.  

You can read more about DistilBERT here:
    https://huggingface.co/docs/transformers/model_doc/distilbert


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""
















# =============================================================================
# NLTK Movie Review dataset
# NLTK = Natural Language ToolKit
# =============================================================================

# import NLTK and Movie Review dataset
import nltk
from nltk.corpus import movie_reviews

# download movie review data
nltk.download('movie_reviews')

# get movie review ids
fileids = movie_reviews.fileids()
len(fileids)  # 2000 movie reviews



# moview review categories
fileids[300]  # negative movie review id 
fileids[1500] # positive movie review id

# OR we can use `.categories()`
movie_reviews.categories(fileids[300])  # negative
movie_reviews.categories(fileids[1500]) # positive

# save all 'categories' into a list 
categories = [movie_reviews.categories(fileid)[0] for fileid in fileids]
categories[300]  # negative
categories[1500] # positive
len(categories)  # 2000 moview review categories



# (raw) movie review contents
movie_reviews.raw(fileids[300])  # negative movie review
movie_reviews.raw(fileids[1500]) # positive movie review

# save all 'reviews' into a list  
reviews = [movie_reviews.raw(fileid) for fileid in fileids]
reviews[300]   # negative review
reviews[1500]  # positive review
len(reviews)   # 2000 reviews














# =============================================================================
# Train & Test datasets
# =============================================================================

# import Numpy & sklearn's train_test_split()  
import numpy as np
from sklearn.model_selection import train_test_split


# labeling 1 as positive category, 0 as negative category
labeling = {'pos': 1, 
            'neg': 0}
y = np.array([labeling[c] for c in categories])
y[300]   # 0, negative category
y[1500]  # 1, positive category


# generate train & test datasets, using train_test_split()
x_train, x_test, y_train, y_test = train_test_split(
    reviews,            # raw moview review contents list from above
    y,                  # 0-1 movie category labeling
    test_size    = 0.2, # let test set be 20% of original dataset
    random_state = 7)


# dataset size
len(x_train)  # 1600
len(x_test)   # 400 = 20% of 2000 (original dataset size)















# =============================================================================
# Classification using pretrained DistilBERT
# =============================================================================

# import AutoClasses 
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# use AutoClasses to use DistilBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
    )
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
    )


# DistilBERT consumes lots of memory so good to divide into (small) batches 
batch_size = 10 # predict 10 each time

# number of batches
num_batch = len(y_test) // batch_size
num_batch # 40 batches

# initiate an empty list of predictions
y_pred = []



# import PyTorch
import torch
import torch.nn.functional as F

# Tokenize data & predict with DistilBERT model
for i in range(num_batch):
    
    # tokenize
    inputs = tokenizer(                        # DistilBERT tokenizer
        x_test[i*batch_size:(i+1)*batch_size], # one batch in each loop
        truncation = True,
        padding    = True,
        return_tensors = "pt"
        )
    
    # use DistilBERT model to predict
    logits = model(**inputs).logits
    
    # apply softmax calculation
    pred = F.softmax(logits, dim = -1)
    
    # get index of max value
    results = pred.detach().numpy().argmax(axis = 1)
    
    # concatenate prediction results to y_pred
    y_pred.extend(results.tolist())
# above for loop takes some time 🕰️ since I am NOT using GPU!



# calculate accuracy
sum(y_test == np.array(y_pred)) / len(y_test) 

'''
84.25% Accuracy!

achieved above accuracy by just using DistilBERT available from Hugging Face!

pretty good accuracy considering 
we did NOT conduct any additional training~
'''





















"""
This is the end of "DistilBERT" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""






















