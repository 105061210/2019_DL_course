#!/usr/bin/env python
# coding: utf-8

# In[1]:


# hw2_105061210
# 因為我是先用 jupyter notebook 寫完才 download 成 .py 檔所以排版可能比較鬆

import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('Data.csv') 


# In[3]:


def Train_Test_Split(data, test_size):
    data_num=data.shape[0]    
    data_index=list(range(data_num))
    test_index=[]
    test_num=int(data_num*test_size)
    for i in range(test_num):
        random_index=int(np.random.uniform(0,len(data_index)))
        test_index.append(data_index[random_index])
        del data_index[random_index]
    train=data.loc[data_index] 
    test=data.loc[test_index]
    return train,test


# In[4]:


train, test = Train_Test_Split(df, 0.2)


# In[5]:


X_train = train.drop('Activities_Types', 1)
X_test = test.drop('Activities_Types', 1)
y_train = train['Activities_Types']
y_test = test['Activities_Types']


# In[6]:


X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)


# In[7]:


y_train_1 = y_train-1
y_test_1 = y_test-1


# In[8]:


import tensorflow as tf


# In[36]:


class DNN(object):
    def __init__(self, n_features, hidden_units=[10], n_classes=2):
        self._n_features = n_features
        self._hidden_units = hidden_units
        self._n_classes = n_classes
        
        self.sess = tf.Session()
    
    def batch_generator(self, X, y, batch_size=128, shuffle=False):
        X_copy = np.array(X)
        y_copy = np.array(y)
        if shuffle:
            data = np.column_stack((X_copy, y_copy))
            np.random.shuffle(data)
            X_copy = data[ : , : -1 ]
            y_copy = data[ : , -1].astype(int)
        
        for i in range(0, X.shape[0], batch_size):
            yield (X.values[i:i+batch_size, : ], y[i:i+batch_size])
    
    def build_model(self):
        hidden = []
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, self._n_features), name='X_input')
        self.y =  tf.placeholder(dtype=tf.int32, shape=None, name='y_input')
        
        with tf.name_scope("input"):
            weights = tf.Variable(tf.truncated_normal(shape = (self._n_features, self._hidden_units[0])), name='weights')
            biases = tf.Variable(tf.zeros(shape = (self._hidden_units[0])), name='biases')
            input_ = tf.matmul(self.X, weights) + biases
            
        
        for index, num_hidden in enumerate(self._hidden_units):
            if index == len(self._hidden_units) - 1: break
            with tf.name_scope("hidden{}".format(index+1)):
                weights = tf.Variable(tf.truncated_normal(shape = (num_hidden, self._hidden_units[index+1])), name='weights')
                biases = tf.Variable(tf.zeros(shape = (self._hidden_units[index+1])), name='biases')
                inputs = input_ if index == 0 else hidden[index-1]
                hidden.append(tf.nn.relu(tf.matmul(inputs, weights) + biases, name="hidden{}".format(index+1)))
        
        with tf.name_scope('output'):
            weights = tf.Variable(tf.truncated_normal(shape = (self._hidden_units[-1], self._n_classes)), name='weights')
            biases = tf.Variable(tf.zeros(shape = (self._n_classes)), name='biases')
            #logits = tf.nn.softmax(tf.matmul(hidden[-1], weights) + biases)
            logits = tf.matmul(hidden[-1], weights) + biases
        return logits
    
    def error(self, y_onehot, logits):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_onehot, logits = logits))
        return cost
    
    def train(self, X_train, X_val, y_train, y_val, epochs=500, lr=0.001, batch_size=128, op=tf.train.AdamOptimizer): 
        #with self.g.as_default():
        self._logits = self.build_model()
        y_onehot = tf.one_hot(indices=self.y, depth=self._n_classes)
        cost = self.error(y_onehot, self._logits)
        optimizer = op(learning_rate=lr)
        train_op = optimizer.minimize(loss=cost)
        
        correct_prediction = tf.equal(tf.argmax(self._logits, 1), tf.argmax(y_onehot, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        init_op = tf.global_variables_initializer() 
        self.sess.run(init_op)
        
        #training_cost = []
        train_loss=[]
        train_acc = []
        val_loss=[]
        val_acc = []
        for i in range(epochs):
            training_cost = []
            batch = self.batch_generator(X_train, y_train, batch_size=batch_size, shuffle=True)
            for batch_X, batch_y in batch:
                _, batch_cost = self.sess.run([train_op, cost], feed_dict={self.X: batch_X, self.y: batch_y})
                training_cost.append(batch_cost)
            train_loss.append(np.mean(training_cost))
            train_acc.append(self.sess.run(accuracy, feed_dict={self.X: X_train, self.y: y_train}))
            val_loss.append(self.sess.run(cost, feed_dict={self.X: X_val, self.y: y_val}))
            val_acc.append(self.sess.run(accuracy, feed_dict={self.X: X_val, self.y: y_val}))
        return train_loss, train_acc, val_loss, val_acc
    
    def predict(self, sample):
        predictions = tf.argmax(self._logits, 1)
        return self.sess.run(predictions, {self.X: sample})


# In[56]:


clf = DNN(n_features=X_train.shape[1], hidden_units=[60, 30], n_classes=6)
train_loss, train_acc, test_loss, test_acc = clf.train(X_train, X_test,y_train_1, y_test_1, epochs=1000, lr=0.001, batch_size=128)


# In[57]:


import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(train_loss, label = 'train_loss')
plt.plot(test_loss, label = 'test_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title("Model Loss")
plt.legend(loc='best')
plt.show()


# In[58]:


plt.figure(figsize=(12, 6))
plt.plot(train_acc, label = 'train_acc')
plt.plot(test_acc, label = 'test_acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title("Model Accuracy")
plt.legend(loc='best')
plt.show()


# In[59]:


# 2. Precision, recall, F1-score
from sklearn.metrics import classification_report
y_pred = clf.predict(X_test)
target_names = ['1', '2', '3', '4', '5', '6']
print('DNN classification_report:')
print(classification_report(y_test_1, y_pred, target_names=target_names))


# In[47]:


# 3. compare with different optimizers
# Gradient descent
clf_gd = DNN(n_features=X_train.shape[1], hidden_units=[60, 30], n_classes=6)
train_loss_gd, train_acc_gd, test_loss_gd, test_acc_gd = clf_gd.train(X_train, X_test, y_train_1, y_test_1, epochs=1000, lr=0.001, batch_size=128, op=tf.train.GradientDescentOptimizer)


# In[48]:


plt.figure(figsize=(12, 6))
plt.plot(train_loss_gd, label = 'train_loss(GD)')
plt.plot(test_loss_gd, label = 'test_loss(GD)')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title("Model Loss")
plt.legend(loc='best')
plt.show()


# In[49]:


plt.figure(figsize=(12, 6))
plt.plot(train_acc_gd, label = 'train_acc(GD)')
plt.plot(test_acc_gd, label = 'test_acc(GD)')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title("Model Accuracy")
plt.legend(loc='best')
plt.show()


# In[44]:


# Adagrad
clf_ag = DNN(n_features=X_train.shape[1], hidden_units=[60, 30], n_classes=6)
train_loss_ag, train_acc_ag, test_loss_ag, test_acc_ag = clf_ag.train(X_train, X_test, y_train_1, y_test_1, epochs=1000, lr=0.01, batch_size=128, op=tf.train.AdagradOptimizer)


# In[45]:


plt.figure(figsize=(12, 6))
plt.plot(train_loss_ag, label = 'train_loss(AG)')
plt.plot(test_loss_ag, label = 'test_loss(AG)')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title("Model Loss")
plt.legend(loc='best')
plt.show()


# In[46]:


plt.figure(figsize=(12, 6))
plt.plot(train_acc_ag, label = 'train_acc(AG)')
plt.plot(test_acc_ag, label = 'test_acc(AG)')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title("Model Accuracy")
plt.legend(loc='best')
plt.show()


# In[20]:


# 4. visualize the validation set through PCA and plot the data using 2 principal components
import sklearn
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_val_pca = pca.fit_transform(X_test)

Xax=X_val_pca[:,0]
Yax=X_val_pca[:,1]
color_dict={1:'red', 2:'orange', 3:'yellow', 4:'green', 5:'blue', 6:'purple'}
name_dict = {1:'dws', 2:'ups', 3:'sit', 4:'std', 5:'wlk', 6:'jog'}
fig,ax=plt.subplots(figsize=(8,6))
for label in np.unique(y_test):
    ix=np.where(label==y_test)
    ax.scatter(Xax[ix], Yax[ix], c=color_dict[label], label=name_dict[label])
    
plt.xlabel("First Principal Component",fontsize=14)
plt.ylabel("Second Principal Component",fontsize=14)
plt.legend()
plt.show()


# In[21]:


# 5. project the validation set onto a 2D space by t-SNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
X_val_tsne = tsne.fit_transform(X_test)

Xax=X_val_tsne[:,0]
Yax=X_val_tsne[:,1]
color_dict={1:'red', 2:'orange', 3:'yellow', 4:'green', 5:'blue', 6:'purple'}
name_dict = {1:'dws', 2:'ups', 3:'sit', 4:'std', 5:'wlk', 6:'jog'}
fig,ax=plt.subplots(figsize=(8,6))
for label in np.unique(y_test):
    ix=np.where(label==y_test)
    ax.scatter(Xax[ix], Yax[ix], c=color_dict[label], label=name_dict[label])

plt.legend()
plt.show()



# In[22]:


# part 2. Hidden Test Set
df_test = pd.read_csv('Test_no_Ac.csv') 


# In[23]:


df_test_pred = clf.predict(df_test)+1


# In[24]:


df_test_pred = pd.DataFrame(df_test_pred)
df_test_pred.to_csv('105061210_answer.txt', header=None, index=True, sep="\t")






