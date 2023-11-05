#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 17:22:48 2023

@author: rj
"""














"""
Sentence Transformer 
 (▰˘◡˘▰)


This video introduces a simple way to compute embeddings for sentences,
using Sentence Transformer available in Hugging Face 🤗

Sentence Transformer can be used to show how "similar" 2 sentences are!




You can read more about Sentence Transformers from the following link:
    https://huggingface.co/docs/hub/sentence-transformers




Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""

















# =============================================================================
# Sentence Similarity
# =============================================================================
'''
You may need to install sentence-transformers first: 
    conda install -c conda-forge sentence-transformers
'''

# import Sentence Transformer
from sentence_transformers import SentenceTransformer, util

# use pretrained sentence transformer model: bert-base-nli-mean-tokens
model = SentenceTransformer('bert-base-nli-mean-tokens')

# I wrote 2 sentences that are similar
sentence1 = "Korea is a beautiful country."
sentence2 = "There are many beautiful places to go in Korea."

# Compute sentence embeddings, using `model.encode()`
sentence1_rep = model.encode(sentence1)
sentence2_rep = model.encode(sentence2)

# Compute Cosine Similarlity, using  pytorch_cos_sim()
# provide sentence1 & sentence2 embeddings; they have "similar" meaning
util.pytorch_cos_sim(sentence1_rep, 
                     sentence2_rep) # 0.8197

# when Cosine Similarity is close to 1, it means 2 sentences are "similar"!


# this time, let's check similarity using a random sentence (sentence3)
sentence3 = "this book is too difficult for me to understand"
sentence3_rep = model.encode(sentence3)
util.pytorch_cos_sim(sentence1_rep, 
                     sentence3_rep) # 0.2584

# Cosine Similarity decreases
# because sentence1 and sentence3 are NOT "similar"

















# =============================================================================
# (Simple) Sentence Similarity Application example
# =============================================================================
'''
Suppose our (imaginary) FAQ website uses Sentence Similarity 
to check if someone has already asked a similar question in the past
'''

# Frequently Asked Questions (FAQs) stored in our imaginary FAQ website
faqs = [
    'what is the exam average score?',
    'when is the exam date?',
    'I want to see my exam score. Where can I check it?',
    'what is the difficulty level of the exam?',
    'How many questions do we need to solve?',
    'what topics do I need to review for the exam?'
    ]

# compute sentence embeddings of our faqs
faqs_rep = model.encode(faqs,
                        convert_to_tensor = True)


# a new user askes the following question in our FAQ website:
query = 'How do I check my exam score?'

# compute sentence embedding of the query
query_rep = model.encode(query,
                         convert_to_tensor = True)


# Compute Cosine Simiarlity 
similarity = util.pytorch_cos_sim(query_rep, faqs_rep)
similarity # looks like 3rd faq earned the highest similarity score

# show the most similar faq, the 3rd faq
import numpy as np
faqs[np.argmax(similarity)]

'''
so when user asks: 'How do I check my exam score?'

our FAQ website found out the following faq is "similar":
    'I want to see my exam score. Where can I check it?'

then our website could display the answer to 3rd faq to our user!
'''
















# =============================================================================
# Multilingual Sentence Transformer
# =============================================================================

# use Multilingual pretrained sentence transformer model
model = SentenceTransformer('distiluse-base-multilingual-cased')

# let's test with 3 languages
english = 'thank you very much'
french  = 'merci beaucoup' # 'thank you' in French
korean  = '정말 감사합니다'    # 'thank you' in Korean
 
# Compute sentence embeddings 
eng_rep = model.encode(english)
fre_rep = model.encode(french)
kor_rep = model.encode(korean)

# check similarity
similarity = util.pytorch_cos_sim(eng_rep, fre_rep) # English & French
similarity # high similarity

similarity = util.pytorch_cos_sim(eng_rep, kor_rep) # English & Korean
similarity # high similarity


# this pretrained model also supports Japanese
japanese = '難しい' # 'difficult' in Japanese
jpn_rep = model.encode(japanese)
similarity = util.pytorch_cos_sim(eng_rep, jpn_rep) # English & Japanese
similarity # low similarity, as expected
























"""
This is the end of "Sentence Transformer" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""





















