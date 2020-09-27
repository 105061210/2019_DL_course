#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2


# In[2]:


# adding noise
def noise(image):
    noise = np.random.randint(5, size = image.shape, dtype='uint8')
    image += noise
    return image


# In[3]:


#flipping
def flip(image):
    image = np.fliplr(image)
    return image


# In[4]:


def shift_left(image):
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            if (i < image.shape[1]-20):
                image[j][i] = image[j][i+20]
            else:
                image[j][i] = 0
    return image


# In[5]:


def shift_right(image):
    for i in range(image.shape[1], 1, -1):
        for j in range(image.shape[0]):
            if (i < image.shape[1]-20):
                image[j][i] = image[j][i-20]
            elif (i < image.shape[1]-1):
                image[j][i] = 0
    return image


# In[6]:


def shift_up(image):
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            if (j < image.shape[0] - 20):
                image[j][i] = image[j+20][i]
            else:
                image[j][i] = 0
    return image


# In[7]:


def shift_down(image):
    for j in range(image.shape[0], 1, -1):
        for i in range(image.shape[1]):
            if (j < image.shape[0] and j > 20):
                image[j][i] = image[j-20][i]
    return image


# In[8]:


def dir_name(file_dir):
    classes = []
    for root, dirs, files in os.walk(file_dir):
        classes.append(dirs)
    return classes[0]


# In[12]:


def preprocess_aug_image(image_path):
    image = cv2.imread(image_path)
    img_n = noise(image)
    img_f = flip(image)
    #img_su = shift_up(image)
    #img_sd = shift_down(image)
    #img_sr = shift_right(image)
    #img_sl = shift_left(image)
    img = cv2.resize(image, (64, 64), cv2.INTER_LINEAR)/255
    img_n = cv2.resize(img_n, (64, 64), cv2.INTER_LINEAR)/255
    img_f = cv2.resize(img_f, (64, 64), cv2.INTER_LINEAR)/255
    #img_su = cv2.resize(img_su, (64, 64), cv2.INTER_LINEAR)/255
    #img_sd = cv2.resize(img_sd, (64, 64), cv2.INTER_LINEAR)/255
    #img_sr = cv2.resize(img_sr, (64, 64), cv2.INTER_LINEAR)/255
    #img_sl = cv2.resize(img_sl, (64, 64), cv2.INTER_LINEAR)/255
    return img, img_n, img_f#, img_su, img_sd, img_sr, img_sl


# In[13]:


def image_aug_label(file_dir, train=True):
    classes = dir_name(file_dir)
    cwd = os.getcwd()
    img_paths = []
    images = []
    labels = []
    index = 0
    for class_name in classes: 
        if train:
            class_path = cwd + '/' + 'train' + '/' + class_name + '/'
        else:
            class_path = cwd + '/' + 'test' + '/' + class_name + '/'
        print('class ', index, '   aug start')
        for img_name in os.listdir(class_path): 
            img_path = class_path + img_name
            img_paths.append(img_path)
            img, img_n, img_f = preprocess_aug_image(img_path)
            images.append(img)
            images.append(img_n)
            images.append(img_f)
            #images.append(img_su)
            #images.append(img_sd)
            #images.append(img_sr)
            #images.append(img_sl)
            labels.append(index)
            labels.append(index)
            labels.append(index)
        index+=1
    print('class ', index, '   aug finish')
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


# In[ ]:


'''
train_path = 'train'
train_image_aug, train_label_aug = image_aug_label(train_path)
'''


# In[ ]:


'''
print(train_image_aug.shape, train_label_aug.shape)
'''

