#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2


# In[3]:


def dir_name(file_dir):
    classes = []
    for root, dirs, files in os.walk(file_dir):
        classes.append(dirs)
    return classes[0]


# In[4]:


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64), cv2.INTER_LINEAR)/255
    return image


# In[5]:


def image_label(classes, train=True):
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
        for img_name in os.listdir(class_path): 
            img_path = class_path + img_name
            img_paths.append(img_path)
            image = preprocess_image(img_path)
            images.append(image)
            labels.append(index)
        index+=1
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


# In[ ]:




