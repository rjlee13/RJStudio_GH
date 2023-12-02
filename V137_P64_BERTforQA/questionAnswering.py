#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 17:22:48 2023

@author: rj
"""














"""
BERT for Question Answering
 (▰˘◡˘▰)

This video introduces BERT Question-Answering model.


Question-Answering tasks consist of 
1) Question and
2) Paragraph that contains Answer  
Question-Answering model is trained to extract Answer from Paragraph.


You can read more about Question-Answering from following link:
https://huggingface.co/docs/transformers/model_doc/bert#transformers


Please 🌟PAUSE🌟 the video any time you want to study the code written.
"""


















# =============================================================================
# Question + Paragraph
# =============================================================================

# Here are my Question + Paragraph
question = 'Which country won Qatar 2022 World Cup?'
paragraph = 'The most recent FIFA World Cup took place in 2022,\
 and Argentina won the tournament.'

'''
Notice Paragraph contains the Answer to my question: Argentina
'''

# For BERT tokenization, 
    # [CLS] is needed at the beginning
    # [SEP] is needed at the end of every sentence
question = '[CLS]' + question + '[SEP]'
paragraph = paragraph + '[SEP]'

# check Question + Paragraph after above (slight) modification 
question  
paragraph
# notice newly added [CLS] & [SEP]




# =============================================================================
# BERT For Question Answering
# =============================================================================

# import BertForQuestionAnswering + BertTokenizer 
from transformers import BertForQuestionAnswering, BertTokenizer


# I chose a pretrained model called:
    # "bert-large-uncased-whole-word-masking-finetuned-squad"
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

'''
you can read more about the pretrained model here:
https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad

SQuAD stands for Stanford Question Answering Dataset
'''



# =============================================================================
# Token & Segment
# =============================================================================

# import PyTorch
import torch

    # Token
# tokenize my Question + Paragraph
q_token = tokenizer.tokenize(question)
p_token = tokenizer.tokenize(paragraph)

# concatenate Question + Paragraph tokens
qp_token = q_token + p_token
qp_token # check tokenized Question + Paragraph

# Convert token string to integer id, using BERT vocabulary
qp_id = tokenizer.convert_tokens_to_ids(qp_token)

# finally, tensorize
qp_id = torch.tensor([qp_id])
qp_id # notice tokens are converted to integer ids


    # Segment
# 0 for question token / 1 for paragraph token
q_segment = [0] * len(q_token) 
p_segment = [1] * len(p_token)

# concatenate 
qp_segment = q_segment + p_segment

# finally, tensorize
qp_segment = torch.tensor([qp_segment])
qp_segment # check




# =============================================================================
# Get Answer!
# =============================================================================

# provide tensorized Token + Segment to BERT model
output = model(qp_id,
               token_type_ids = qp_segment)

# start index of Answer
answer_start_index = output.start_logits.argmax()
# end index of Answer
answer_end_index = output.end_logits.argmax()

# Get Answer using start & end indices
print(''.join(qp_token[answer_start_index:answer_end_index+1]))
# Yes, Argentina won Qatar 2022 World Cup!  






















"""
This is the end of "BERT for Question Answering" video~


Hope you enjoyed it!
Thank you for watching ◎[▪‿▪]◎ 
"""





















