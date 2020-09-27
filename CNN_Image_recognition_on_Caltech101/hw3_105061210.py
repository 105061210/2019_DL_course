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


from data_prepro import dir_name, image_label


# In[3]:


#implement a CNN for image recognition
class CNN(object):
    def __init__(self, batch_size=256, epochs=30, learning_rate=0.0005, dropout=0.5, shuffle=True):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.shuffle = shuffle
        self.build()
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("TensorBoard/", graph = self.sess.graph)
    
    def conv_layer(self, input_tensor, name, kernel_size, n_output, padding='SAME', strides = (1,1,1,1), activate = tf.nn.relu):
        with tf.variable_scope(name):
            input_shape = input_tensor.get_shape().as_list()
            n_input = input_shape[-1]
            weight_shape = (list(kernel_size) + [n_input, n_output])
            weight = tf.get_variable(name = 'weight', shape = weight_shape)
            tf.summary.histogram(name = name + '/weights', values = weight)
            bias = tf.get_variable(name = 'bias', initializer = tf.zeros(shape = [n_output]))
            conv = tf.nn.conv2d(input = input_tensor, filter = weight, strides = strides, padding = padding)
            conv = tf.nn.bias_add(conv, bias, name = 'pre_activation')
            conv = activate(conv, name = 'activation')
            return conv
        
    def fc_layer(self, input_tensor, name, n_output, activate = None):
        with tf.variable_scope(name):
            input_shape = input_tensor.get_shape().as_list()[1:]
            n_input = np.prod(input_shape)
            if len(input_shape) > 1:
                input_tensor = tf.reshape(input_tensor, shape = (-1, n_input))
            weight_shape = [n_input, n_output]
            weight = tf.get_variable(name = 'weight', shape = weight_shape)
            tf.summary.histogram(name = name + '/weights', values = weight)
            bias = tf.get_variable(name = 'bias', initializer = tf.zeros(shape = [n_output]))
            fc = tf.matmul(input_tensor, weight)
            fc = tf.nn.bias_add(fc, bias, name = 'pre_activation')
            if activate is None:
                return fc
            fc = activate(fc, name = 'activation')
            return fc
        
    def build(self):
        tf_y = tf.placeholder(tf.int32, shape = [None], name = 'tf_y')
        
        tf_x_image = tf.placeholder(tf.float32, shape = [None, 64, 64, 3], name = 'tf_x_image')
        tf_y_onehot = tf.one_hot(indices = tf_y, depth = 101, dtype = tf.float32, name = 'tf_y_onehot')
        
        self.h1 = self.conv_layer(tf_x_image, name = 'conv_1', kernel_size = (5, 5), padding = 'VALID', n_output = 32)
        h1_pool = tf.nn.max_pool(self.h1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        
        self.h2 = self.conv_layer(h1_pool, name = 'conv_2', kernel_size = (5, 5), padding = 'VALID', n_output = 64)
        h2_pool = tf.nn.max_pool(self.h2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        
        self.h3 = self.conv_layer(h2_pool, name = 'conv_3', kernel_size = (5, 5), padding = 'VALID', n_output = 128)
        h3_pool = tf.nn.max_pool(self.h3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        
        h4 = self.fc_layer(h3_pool, name = 'fc_4', n_output = 1024, activate = tf.nn.relu)
        keep_prob = tf.placeholder(tf.float32, name = 'fc_keep_prob')
        h4_drop = tf.nn.dropout(h4, keep_prob = keep_prob, name = 'fc_4_drop')
        
        h5 = self.fc_layer(h4_drop, name = 'fc_5', n_output = 101, activate = None)
        
        pred = {
            'prob': tf.nn.softmax(h5, name = 'prob'),
            'label': tf.cast(tf.argmax(h5, axis = 1), tf.int32, name = 'label')
        }
        
        celoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=h5, labels=tf_y_onehot), name = 'celoss')
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(celoss, name = 'train_op')
        
        correct_pred = tf.equal(pred['label'], tf_y, name = 'correct_pred')
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = 'accuracy')
        print('build successfully')
        
    def train(self, X_train, X_val, y_train, y_val, init = True):
        if init:
            self.sess.run(self.init_op)
            print('initialize successfully')
        #self.build()
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        
        for epoch in range(1, self.epochs+1):
            batch_cost = []
            batch = self.batch_generator(X_train, y_train, batch_size=self.batch_size, shuffle=True, i=epoch)
            print('batch generated for epoch ', epoch)
            #i=0
            for batch_X, batch_y in batch:
                _, cost = self.sess.run(['train_op', 'celoss:0'], feed_dict={'tf_x_image:0': batch_X, 'tf_y:0': batch_y, 'fc_keep_prob:0': self.dropout})
                batch_cost.append(cost)
                #print('batch ', i, '   cost = ', cost)
                #i=i+1
            self.train_loss.append(np.mean(batch_cost))
            print('epoch: ', epoch, '    train loss: ', self.train_loss[epoch-1])
            self.train_acc.append(self.sess.run('accuracy:0', feed_dict={'tf_x_image:0': X_train, 'tf_y:0': y_train, 'fc_keep_prob:0': self.dropout}))
            print('epoch: ', epoch, '    train accuracy: ', self.train_acc[epoch-1])
            self.val_loss.append(self.sess.run('celoss:0', feed_dict={'tf_x_image:0': X_val, 'tf_y:0': y_val, 'fc_keep_prob:0': 1.0}))
            print('epoch: ', epoch, '    val loss: ', self.val_loss[epoch-1])
            self.val_acc.append(self.sess.run('accuracy:0', feed_dict={'tf_x_image:0': X_val, 'tf_y:0': y_val, 'fc_keep_prob:0': 1.0}))
            print('epoch: ', epoch, '    val accuracy: ', self.val_acc[epoch-1])
            print('--------------------------------------------------------')
        return self.train_loss, self.train_acc, self.val_loss, self.val_acc
    
    def batch_generator(self, X, y, batch_size = 256, shuffle = True, i=123):
        idx = np.arange(y.shape[0])
        if shuffle:
            rng = np.random.RandomState(i)
            rng.shuffle(idx)
            X = X[idx]
            y = y[idx]
        for i in range(0, X.shape[0], batch_size):
            yield(X[i:i+batch_size, : ], y[i:i+batch_size])
            
    def write(self, X, y):
        result = self.sess.run(self.merged, feed_dict={'tf_x_image:0': X, 'tf_y:0': y, 'fc_keep_prob:0': 1.0})
        self.writer.add_summary(result)
        
    def predict(self, X, proba = False):
        feed_dict={'tf_x_image:0': X, 'fc_keep_prob:0': 1.0}
        if proba:
            return self.sess.run('prob:0', feed_dict = feed_dict)
        else:
            return self.sess.run('label:0', feed_dict = feed_dict)
    
    def print_conv1(self, X):
        return self.sess.run(self.h1, feed_dict={'tf_x_image:0': X, 'fc_keep_prob:0': 1.0})
    
    def print_conv2(self, X):
        return self.sess.run(self.h2, feed_dict={'tf_x_image:0': X, 'fc_keep_prob:0': 1.0})
    


# In[4]:


cwd = os.getcwd()
train_path = 'train' 
test_path = 'test'
train_classes = dir_name(train_path)
test_classes = dir_name(test_path)
print(len(train_classes), train_classes)
print(len(test_classes), test_classes)
label_dict = { i: train_classes[i] for i in range(0, len(train_classes) ) }


# In[5]:


train_image, train_label = image_label(train_classes)
print(train_image.shape, train_label.shape)


# In[6]:


test_image, test_label = image_label(test_classes, train = False)
print(test_image.shape, test_label.shape)


# In[7]:


clf = CNN()


# In[8]:


train_loss, train_acc, val_loss, val_acc = clf.train(train_image, test_image, train_label, test_label)


# In[9]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot(train_loss, label = 'training loss')
plt.plot(val_loss, label = 'test loss')
plt.xlabel('epoch')
plt.ylabel('Cross entropy')
plt.title("Learning curve")
plt.legend(loc='best')
plt.show()


# In[10]:


plt.figure(figsize=(8, 6))
plt.plot(train_acc, label = 'training acc')
plt.plot(val_acc, label = 'test acc')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.title("Model Accuracy")
plt.legend(loc='best')
plt.show()


# In[11]:


clf.write(train_image, train_label)


# In[12]:


pred = clf.predict(test_image[1030:1031])
print('predict: ', label_dict[pred[0]], '    true label: ', label_dict[test_label[1030]])


# In[13]:


def plot_filter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(16,16))
    n_columns = 8
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i])


# In[14]:


conv1 = clf.print_conv1(test_image[1030:1031])
plot_filter(conv1)


# In[15]:


conv2 = clf.print_conv2(test_image[1030:1031])
plot_filter(conv2)


# In[20]:


plt.title('predict: '+ label_dict[pred[0]] + '  true label: ' + label_dict[test_label[1030]])
plt.imshow(test_image[1030])


# In[ ]:




