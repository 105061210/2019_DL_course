#!/usr/bin/env python
# coding: utf-8

#因為我是從jupyter notebook上載下來成為.py檔，所以排版可能比較鬆

# In[1]:


import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import math


# In[2]:


from data_augmentation import image_aug_label
from data_prepro import image_label, dir_name


# In[3]:


cwd = os.getcwd()
train_path = 'train' 
test_path = 'test'
train_classes = dir_name(train_path)
test_classes = dir_name(test_path)
#print(len(train_classes), train_classes)
#print(len(test_classes), test_classes)
label_dict = { i: train_classes[i] for i in range(0, len(train_classes) ) }


# In[4]:


train_image, train_label = image_aug_label(train_path)
print(train_image.shape, train_label.shape)


# In[5]:


test_image, test_label = image_label(test_classes, train = False)
print(test_image.shape, test_label.shape)


# In[6]:



def batch_generator(X, y, batch_size = 512, shuffle = True, i=123):
    idx = np.arange(y.shape[0])
    if shuffle:
        rng = np.random.RandomState(i)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]
    for i in range(0, X.shape[0], batch_size):
        yield(X[i:i+batch_size, : ], y[i:i+batch_size])

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


x_image = tf.placeholder(tf.float32, [None, 64, 64, 3])
ys = tf.placeholder(tf.int32, [None])
tf_y_onehot = tf.one_hot(indices = ys, depth = 101, dtype = tf.float32)
keep_prob = tf.placeholder(tf.float32)



## conv1 layer ##
W_conv1 = tf.get_variable(name = 'W_conv1', shape = [5, 5, 3, 32]) # patch 5x5, in size 1, out size 32
b_conv1 = tf.get_variable(name = 'b_conv1', initializer = tf.zeros(shape = [32]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='VALID') + b_conv1) 
h_pool1 =  tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')   

## conv2 layer ##
W_conv2 = tf.get_variable(name = 'W_conv2', shape = [5, 5, 32, 64]) # patch 5x5, in size 32, out size 64
b_conv2 = tf.get_variable(name = 'b_conv2', initializer = tf.zeros(shape = [64]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='VALID') + b_conv2) 
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')   

## conv3 layer ##
W_conv3 = tf.get_variable(name = 'W_conv3', shape = [5, 5, 64, 128]) # patch 5x5, in size 32, out size 64
b_conv3 = tf.get_variable(name = 'b_conv3', initializer = tf.zeros(shape = [128]))
h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='VALID') + b_conv3) 
h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  


## fc1 layer ##
W_fc1 = tf.get_variable(name = 'W_fc1', shape = [3200, 1024])
b_fc1 = tf.get_variable(name = 'b_fc1', initializer = tf.zeros(shape = [1024]))
input_shape = h_pool3.get_shape().as_list()[1:]
n_input = np.prod(input_shape)
#print(n_input)
if len(input_shape) > 1:
    h_pool3_flat = tf.reshape(h_pool3, [-1, n_input])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 =  tf.get_variable(name = 'W_fc2', shape = [1024, 101])
b_fc2 = tf.get_variable(name = 'b_fc2', initializer = tf.zeros(shape = [101]))
output = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

pred = {
        'prob': tf.nn.softmax(output),
        'label': tf.cast(tf.argmax(output, axis = 1), tf.int32)
}

celoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=tf_y_onehot))
regularizer = tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_fc2)
reg_celoss = (celoss + 0.0005 * regularizer)
optimizer = tf.train.AdamOptimizer(0.0002)
train_step = optimizer.minimize(reg_celoss)

correct_pred = tf.equal(pred['label'], ys)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

train_loss = []
train_acc = []
val_loss = []
val_acc = []

for epoch in range(1, 30+1):
    batch_cost = []
    batch_acc = []
    batch = batch_generator(train_image, train_label, i=epoch)
    print('batch generated for epoch ', epoch)
    for batch_X, batch_y in batch:
        _, cost, acc = sess.run([train_step, reg_celoss, accuracy], feed_dict={x_image: batch_X, ys: batch_y, keep_prob: 0.5})
        batch_cost.append(cost)
        batch_acc.append(acc)
    train_loss.append(np.mean(batch_cost))
    train_acc.append(np.mean(batch_acc))
    print('epoch: ', epoch, '    train loss: ', train_loss[epoch-1])
    print('epoch: ', epoch, '    train accuracy: ', train_acc[epoch-1])
    val_loss.append(sess.run(reg_celoss, feed_dict={x_image: test_image, ys: test_label,  keep_prob: 1.0}))
    print('epoch: ', epoch, '    val loss: ', val_loss[epoch-1])
    val_acc.append(sess.run(accuracy, feed_dict={x_image: test_image, ys: test_label,  keep_prob: 1.0}))
    print('epoch: ', epoch, '    val accuracy: ', val_acc[epoch-1])
    print('--------------------------------------------------------')


# In[7]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot(train_loss, label = 'training loss')
plt.plot(val_loss, label = 'test loss')
plt.xlabel('epoch')
plt.ylabel('Cross entropy')
plt.title("Learning curve")
plt.legend(loc='best')
plt.show()


# In[8]:


plt.figure(figsize=(8, 6))
plt.plot(train_acc, label = 'training acc')
plt.plot(val_acc, label = 'test acc')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.title("Model Accuracy")
plt.legend(loc='best')
plt.show()


# In[9]:


w_conv1 = W_conv1.eval(session=sess)
w_conv1_re = w_conv1.reshape(2400)
plt.title('distribution of weights of conv1')
plt.hist(w_conv1_re)
plt.show()


# In[10]:


W_fc2.get_shape()


# In[11]:


w_conv2 = W_conv2.eval(session=sess)
w_conv2_re = w_conv2.reshape(5*5*32*64)
plt.title('distribution of weights of conv2')
plt.hist(w_conv2_re)
plt.show()


# In[12]:


w_conv3 = W_conv3.eval(session=sess)
w_conv3_re = w_conv3.reshape(5*5*64*128)
plt.title('distribution of weights of conv3')
plt.hist(w_conv3_re)
plt.show()


# In[13]:


w_fc1 = W_fc1.eval(session=sess)
w_fc1_re = w_fc1.reshape(3200*1024)
plt.title('distribution of weights of fc1')
plt.hist(w_fc1_re)
plt.show()


# In[14]:


w_fc2 = W_fc2.eval(session=sess)
w_fc2_re = w_fc2.reshape(1024*101)
plt.title('distribution of weights of fc2')
plt.hist(w_fc2_re)
plt.show()


# In[ ]:




