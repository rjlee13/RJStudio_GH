#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 17:22:48 2023

@author: rj
"""










"""
Introduction to Hugging Face
 (▰˘◡˘▰)


Hugging Face is a very popular Machine Learning / AI community~
There are so many AI tasks you can accomplish using various models in HF.
But it is particularly useful for NLP tasks!




Take a look! ->  https://huggingface.co/




This video explains very simple ways to start using Hugging Face 🤗

Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""

















# =============================================================================
# pipeline
# =============================================================================
'''
A very 'very' simple way to use Hugging Face transformer is to 
import pipeline


you may need to execute following commands first:
    pip install transformers
    conda install -c huggingface transformers
'''

# import pipeline 
from transformers import pipeline




# Let's try Sentiment Analysis task! 
sentiment_analyzer = pipeline("sentiment-analysis")
# using 'distilbert-base-uncased-finetuned-sst-2-english'

sentiment_analyzer("That was really good")          # POSITIVE 99% score
sentiment_analyzer("It was boring... I fell asleep")# NEGATIVE 99% score
sentiment_analyzer("debatable.. okay I guess")      # NEGATIVE 86% score


# conduct Sentiment Analysis with more than 1 sentence
sentiment_analyzer([
    "fantastic",     # POSITIVE 99%
    "terrible"       # NEGATIVE 99%
    ]) 




# Let's try Text Generation task!  
text_generator = pipeline("text-generation")
# using 'gpt2'

# provide a phrase to start with 
text_generator("I woke up early ")
# if you execute the above command AGAIN, you see a DIFFERENT result





# Translation : English to French
translator = pipeline("translation_en_to_fr")
# using 't5-base'

translator("hello")           # Bonjour
translator("my name is Mike") # mon nom est Mike



'''
using pipeline, you can also perform other simple tasks:
    text classification
    question answering
    summarization

try them out~ 🫶
'''














# =============================================================================
# AutoClasses
# =============================================================================
'''
Autoclasses 'automatically' choose the relevant model for you 
if you provide the name of the pretrained model 🌟
'''

# import AutoTokenizer, AutoModel, Pytorch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

'''
I want to conduct the following task:
    is 'target_sentence' paraphrasing 'input_sentence' ??
'''

# Here are my input & target sentences: 
input_sentence = "By tomorrow, I need to finish up my project \
    and study for an important exam"
target_sentence = "I have to complete my project and study"



# Use AutoClasses and provide 'bert-base-cased-finetuned-mrpc' model
    # mrpc = Microsoft Research Paraphrase Corpus

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased-finetuned-mrpc")

# AutoClasses has picked appropriate tokenizer and model for you



# tokenize input_sentence & target_sentence
tokens = tokenizer(input_sentence, 
                   target_sentence, 
                   return_tensors = "pt")

# use my model to check paraphrase! 
logits  = model(**tokens).logits                   # before softmax calculation
results = torch.softmax(logits, dim=1).tolist()[0] # apply  softmax calculation
for i, label in enumerate(['no','yes']):
    print(f'{label}:{int(round(results[i]*100))}%')

# model says "yes" with high probability
# in other words, target_sentence is likely paraphrasing input_sentence ✅


    

# Test by giving IRRELEVANT bad_target ❌
# rest of the code is the SAME
bad_target = "I want to watch a movie"
tokens = tokenizer(input_sentence, 
                   bad_target,             # supply bad_target this time
                   return_tensors = "pt")
logits = model(**tokens).logits
results = torch.softmax(logits, dim=1).tolist()[0]
for i, label in enumerate(['no','yes']):
    print(f'{label}:{int(round(results[i]*100))}%')

# this time, model says "no" with high probability
# in other words, bad_target is probably NOT paraphrasing, expected result!

























"""
This is the end of "Intro to Hugging Face" video~

This video covered 2 topics:
    pipeline
    AutoClasses


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""































# =============================================================================
# 
# =============================================================================

import nltk
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
import numpy as np


nltk.download('movie_reviews')

fileids = movie_reviews.fileids()
fileids

movie_reviews.raw('neg/cv997_5152.txt')
reviews = [movie_reviews.raw(fileid) for fileid in fileids]
reviews[0]

movie_reviews.categories('neg/cv997_5152.txt')
categories = [movie_reviews.categories(fileid)[0] for fileid in fileids]
categories[0]


label_dict = {'pos':1, 
              'neg': 0}

y = np.array([label_dict[c] for c in categories])


x_train, x_test, y_train, y_test = train_test_split(
    reviews, y, 
    test_size = 0.2,
    random_state = 7)


x_train[0]


len(x_train)
len(x_test)




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
    )
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
    )


model = model.to(device)

# number of data to predict each time
batch_size = 10

# list for saving predictions
y_pred = []

num_batch = len(y_test) // batch_size
num_batch # 40

import torch.nn.functional as F

for i in range(num_batch):
    inputs = tokenizer(
        x_test[i*batch_size:(i+1)*batch_size],
        truncation = True,
        padding = True,
        return_tensors = "pt"
        )

    inputs = inputs.to(device)
    
    
    
    logits = model(**inputs).logits
    
    
    
    pred = F.softmax(logits, dim = -1)
    
    results = pred.cpu().detach().numpy().argmax(axis = 1)
    y_pred.extend(results.tolist())


y_test
sum(y_test == np.array(y_pred)) / len(y_test)























