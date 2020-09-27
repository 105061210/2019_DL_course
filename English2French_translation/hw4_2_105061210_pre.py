#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pickle
import copy
from collections import Counter


# In[2]:


def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()
    return data


# In[3]:


source_path = 'en.txt'
target_path = 'fr.txt'
source_text = load_data(source_path)    #english
target_text = load_data(target_path)    #french


# In[4]:


print('* number of unique words in English sample sentences: {}'.format(len(Counter(source_text.split()))))

english_sentences = source_text.split('\n')
print('* English sentences')
print('\t- number of sentences: {}'.format(len(english_sentences)))
print('\t- average number of words in a sentence: {}'.format(np.average([len(sentence.split()) for sentence in english_sentences])))

french_sentences = target_text.split('\n')
print('* French sentences')
print('\t- number of sentences: {}'.format(len(french_sentences)))
print('\t- average number of words in a sentence: {}'.format(np.average([len(sentence.split()) for sentence in french_sentences])))
print()
print(english_sentences[0])
print(french_sentences[0])
print()
print(english_sentences[1])
print(french_sentences[1])


# In[5]:


special_tokens = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3 }
def word_int_tables(text):
    unique_word = set(text.split())
    vocab_to_int = copy.copy(special_tokens)
    for i_i, i in enumerate(unique_word, len(special_tokens)):
        vocab_to_int[i] = i_i
        
    int_to_vocab = {i_i: i for i, i_i in vocab_to_int.items()}

    return vocab_to_int, int_to_vocab


# In[6]:


def transfer_text2index(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    source_text_id = []
    target_text_id = []
    
    source_sentences = source_text.split("\n")
    target_sentences = target_text.split("\n")
    
    max_source_sentence_length = max([len(sentence.split(" ")) for sentence in source_sentences])
    max_target_sentence_length = max([len(sentence.split(" ")) for sentence in target_sentences])
    
    for i in range(len(source_sentences)):
        source_sentence = source_sentences[i]
        target_sentence = target_sentences[i]
        
        source_tokens = source_sentence.split(" ")
        target_tokens = target_sentence.split(" ")
        
        source_token_id = []
        target_token_id = []
        
        for index, token in enumerate(source_tokens):
            if (token != ""):
                source_token_id.append(source_vocab_to_int[token])
        
        for index, token in enumerate(target_tokens):
            if (token != ""):
                target_token_id.append(target_vocab_to_int[token])
                
        target_token_id.append(target_vocab_to_int['<EOS>'])
        source_text_id.append(source_token_id)
        target_text_id.append(target_token_id)
    
    return source_text_id, target_text_id


# In[7]:


source_vocab_to_int, source_int_to_vocab = word_int_tables(source_text)
target_vocab_to_int, target_int_to_vocab = word_int_tables(target_text)


# In[8]:


source_text_id, target_text_id = transfer_text2index(source_text, target_text, source_vocab_to_int, target_vocab_to_int)


# In[9]:


pickle.dump((
        (source_text_id, target_text_id),
        (source_vocab_to_int, target_vocab_to_int),
        (source_int_to_vocab, target_int_to_vocab)), open('preprocess.p', 'wb'))


# In[ ]:




