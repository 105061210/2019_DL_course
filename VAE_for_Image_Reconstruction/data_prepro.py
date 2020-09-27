#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# In[2]:


#we only want char A to Z, which are classes 10 to 35
def load_sortChar():
    data = pd.read_csv('emnist-balanced-train.csv', header=None)
    #data.sort_values(by = [0], inplace=True)
    data_char = data[data[0]>9]
    data_char = data_char[data_char[0]<36]
    return data_char


# In[53]:


def preprocess():
    data_char = load_sortChar()
    imgs = np.transpose(data_char.values[:,1:].reshape(len(data_char), 28, 28), axes=[0, 2, 1])
    img_noise =imgs+40*np.random.randn(len(data_char), 28, 28)                
    imgs = imgs/255
    img_noise = img_noise/255 
    data = imgs.reshape(len(data_char), 784)
    data_noise = img_noise.reshape(len(data_char), 784)
    labels = data_char.values[:,0]
    return imgs, img_noise, data, data_noise, labels


# In[4]:


def print_img(vector):
    img = np.transpose(vector.reshape(28, 28), axes = [0,1])
    plt.imshow(img , cmap='Greys_r')


# In[5]:


def label2char(label):
    label2digit_char = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
    return label2digit_char[label]






