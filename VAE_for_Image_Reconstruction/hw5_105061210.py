#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


# In[2]:


from data_prepro import preprocess, print_img, label2char


# In[3]:


imgs, img_noise, data, data_noise, labels = preprocess()


# In[4]:


print(imgs.shape)
print(img_noise.shape)
print(data.shape)
print(data_noise.shape)
print(labels.shape)


# In[5]:


class VAE(object):
    def __init__(self, latent_dim = 2, lr = 0.0001, batch_size = 200):
        self.latent_dim = latent_dim
        self.lr = lr
        self.batch_size = batch_size
        #tf.reset_default_graph()
        self.g = tf.Graph()
        with self.g.as_default():
            self.build()
            self.init_op = tf.global_variables_initializer()
        self.sess = tf.Session(graph = self.g)
    
    def build(self):
        self.tf_x = tf.placeholder(tf.float32, shape = [None, 784], name = 'tf_x')  #for input data(with noise)
        self.tf_t = tf.placeholder(tf.float32, shape = [None, 784], name = 'tf_t')  #for target 
        keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
        
        
        with tf.variable_scope("encoder"):
            en_w_init = tf.contrib.layers.variance_scaling_initializer()
            en_b_init = tf.constant_initializer(0.)
            
            # 1st hidden layer
            en_w1 = tf.get_variable('en_w1', [784, 392], initializer=en_w_init)
            en_b1 = tf.get_variable('en_b1', [392], initializer=en_b_init)
            en_h1 = tf.matmul(self.tf_x, en_w1) + en_b1
            en_h1 = tf.nn.relu(en_h1)
            en_h1 = tf.nn.dropout(en_h1, keep_prob)
            print(en_h1)
            
            # 2nd hidden layer
            en_w2 = tf.get_variable('en_w2', [en_h1.get_shape()[1], 196], initializer=en_w_init)
            en_b2 = tf.get_variable('en_b2', [196], initializer=en_b_init)
            en_h2 = tf.matmul(en_h1, en_w2) + en_b2
            en_h2 = tf.nn.relu(en_h2)
            en_h2 = tf.nn.dropout(en_h2, keep_prob)
            print(en_h2)
            
            # 3rd hidden layer
            en_w3 = tf.get_variable('en_w3', [en_h2.get_shape()[1], 100], initializer=en_w_init)
            en_b3 = tf.get_variable('en_b3', [100], initializer=en_b_init)
            en_h3 = tf.matmul(en_h2, en_w3) + en_b3
            en_h3 = tf.nn.relu(en_h3)
            en_h3 = tf.nn.dropout(en_h3, keep_prob)
            print(en_h3)
            
            #output layer
            en_wout = tf.get_variable('en_wout', [en_h3.get_shape()[1], self.latent_dim], initializer=en_w_init)
            en_bout = tf.get_variable('en_bout', [self.latent_dim], initializer=en_b_init)
        
        self.mu = tf.matmul(en_h3, en_wout) + en_bout
        self.sigma = tf.matmul(en_h3, en_wout) + en_bout
            
        eps = tf.random_normal(tf.shape(self.sigma),mean=0, stddev=1, dtype=tf.float32)
        self.z = self.mu + tf.sqrt(tf.exp(self.sigma)) * eps
        print(self.z)
        
        with tf.variable_scope("decoder"):
            de_w_init = tf.contrib.layers.variance_scaling_initializer()
            de_b_init = tf.constant_initializer(0.)
            
            # 1st hidden layer
            de_w1 = tf.get_variable('de_w1', [self.z.get_shape()[1], 100], initializer=de_w_init)
            de_b1 = tf.get_variable('de_b0', [100], initializer=de_b_init)
            de_h1 = tf.matmul(self.z, de_w1) + de_b1
            de_h1 = tf.nn.relu(de_h1)
            de_h1 = tf.nn.dropout(de_h1, keep_prob)
            print(de_h1)

            # 2nd hidden layer
            de_w2 = tf.get_variable('de_w2', [de_h1.get_shape()[1], 196], initializer=de_w_init)
            de_b2 = tf.get_variable('de_b2', [196], initializer=de_b_init)
            de_h2 = tf.matmul(de_h1, de_w2) + de_b2
            de_h2 = tf.nn.relu(de_h2)
            de_h2 = tf.nn.dropout(de_h2, keep_prob)
            print(de_h2)
            
            # 3rd hidden layer
            de_w3 = tf.get_variable('de_w3', [de_h2.get_shape()[1], 392], initializer=de_w_init)
            de_b3 = tf.get_variable('de_b3', [392], initializer=de_b_init)
            de_h3 = tf.matmul(de_h2, de_w3) + de_b3
            de_h3 = tf.nn.relu(de_h3)
            de_h3 = tf.nn.dropout(de_h3, keep_prob)
            print(de_h3)

            # output layer
            de_wout = tf.get_variable('de_wout', [de_h3.get_shape()[1], 784], initializer=de_w_init)
            de_bout = tf.get_variable('de_bout', [784], initializer=de_b_init)
            self.y = tf.sigmoid(tf.matmul(de_h3, de_wout) + de_bout)
            print(self.y)
        
        epsilon = 1e-10
        self.reconstruction_loss = -tf.reduce_sum(self.tf_t * tf.log(epsilon+self.y) + (1.0 - self.tf_t) * tf.log(1.0 - self.y + epsilon), 1)
        self.reconstruction_loss = tf.reduce_mean(self.reconstruction_loss)
        self.KL_divergence = -0.5 * tf.reduce_sum(1 + self.sigma - tf.square(self.mu) - tf.exp(self.sigma), 1)
        self.KL_divergence = tf.reduce_mean(self.KL_divergence)
        reg = tf.contrib.layers.l2_regularizer(scale=0.05)
        self.reg_loss = tf.contrib.layers.apply_regularization(reg, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))   #L2 regularization
        
        total_loss = self.reconstruction_loss + self.KL_divergence
        self.loss = total_loss +self.reg_loss
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        
        
    def batch_generator(self, X, y, batch_size = 256, shuffle = True, i=123):
        idx = np.arange(y.shape[0])
        if shuffle:
            rng = np.random.RandomState(i)
            rng.shuffle(idx)
            X = X[idx]
            y = y[idx]
        for i in range(0, X.shape[0], batch_size):
            yield(X[i:i+batch_size, : ], y[i:i+batch_size, : ])
    
    def train(self, X, y, epochs):
        self.sess.run(self.init_op)
        self.train_cost = []
        for epoch in range(1, epochs+1):
            batch_cost = []
            batch_regs = []
            batch = self.batch_generator(X, y, batch_size=self.batch_size, shuffle=True, i=epoch)
            for batch_X, batch_y in batch:
                _, batch_loss, batch_reg, output = self.sess.run([self.train_op, self.loss, self.reconstruction_loss, self.y], feed_dict={'tf_x:0': batch_X, 'tf_t:0': batch_y, 'keep_prob:0':0.9})
                batch_cost.append(batch_loss)
                batch_regs.append(batch_reg)
            self.train_cost.append(np.mean(batch_cost))
            print('epoch: ', epoch, '    train loss: {:.3f}'.format(self.train_cost[epoch-1]), '    reconstruction: ', np.mean(batch_regs))
        return self.train_cost, output
    
    def reconstructor(self, x):    #from input image to generate output image
        y = self.sess.run(self.y, feed_dict={'tf_x:0': x, 'keep_prob:0': 1.0})
        return y

    
    def generator(self, z):    #from latent vector to generate output image
        y = self.sess.run(self.y, feed_dict={self.z: z, 'keep_prob:0': 1.0})
        return y
    
    
    def transformer(self, x):    #encode input image to vector
        z = self.sess.run(self.z, feed_dict={'tf_x:0': x, 'keep_prob:0': 1.0})
        return z


# In[6]:


#for plot 3*3=9 image (question 4.)
def plot_img(X, label): 
    plt.figure(figsize=(4, 4))
    for i in range(len(X)):
        img = np.transpose(X[i].reshape(28, 28), axes = [0,1])
        plt.subplot(3, 3, i+1)
        plt.title('Char ' + label2char(label[i]))
        plt.axis('off')
        plt.imshow(img, cmap='gray')


# In[7]:


#for plot reconstruction results
def plot_reconstructor(X, model):
    y = model.reconstructor(X)
    n = np.sqrt(len(y)).astype(np.int32)
    img = np.empty((28*n, 28*n))
    for i in range(n):
        for j in range(n):
            y_ = y[i*n+j, :].reshape(28, 28)
            img[i*28:(i+1)*28, j*28:(j+1)*28] = y_
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')   


# In[8]:


#plot the images that generated from latent vectors
def plot_generator(X, model):
    y = model.generator(X)
    n = np.sqrt(len(y)).astype(np.int32)
    img = np.empty((28*n, 28*n))
    for i in range(n):
        for j in range(n):
            y_ = y[i*n+j, :].reshape(28, 28)
            img[i*28:(i+1)*28, j*28:(j+1)*28] = y_
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')   


# In[9]:


#plot original input data
plot_img(data[47:56], labels[47:56])


# In[10]:


#plot data with adding noise
plot_img(data_noise[47:56], labels[47:56])


# In[11]:


#VAE with latent vector dimension=2, batch size=200
vae_2 = VAE(latent_dim = 2, lr = 0.003, batch_size = 200)


# In[13]:


loss_2, out2 = vae_2.train(data_noise, data, 50)


# In[14]:


plt.plot(- np.array(loss_2)/784/200)
plt.title('Latent dimension = 2 (batch_size = 200)')


# In[15]:


#show some examples reconstructed from the decoder.
plot_img(vae_2.reconstructor(data[47:56]), labels[47:56])


# In[16]:


#transform data into latent vector
vector_2 = vae_2.transformer(data[6:7])

#plot the reconstructed images by varying the value of the latent variable.
from copy import deepcopy
vec_2 = []
for i in range(20):
    for j in range(20):
        vec = deepcopy(vector_2)
        vec[0, 0] += i*0.5
        vec[0, 1] += j*0.5
        vec_2.append(vec)

plot_generator(np.reshape(np.array(vec_2), (400,2)), vae_2)


# In[17]:


#Sample noise vectors from N(0,1) as latent variables and use the decoder to generate some images.
vector_for_2 = 1.5*np.random.randn(100,2)
plot_generator(vector_for_2, vae_2)


# In[ ]:





# In[18]:


#VAE with latent vector dimension=10, batch size=200
vae_10 = VAE(latent_dim = 10, lr = 0.003, batch_size = 200)
loss_10, out10 = vae_10.train(data_noise, data, 50)


# In[19]:


plt.plot(-np.array(loss_10)/784/200)
plt.title('Latent dimension = 10 (batch_size = 200)')


# In[20]:


#show some examples reconstructed from the decoder.
plot_img(vae_10.reconstructor(data[47:56]), labels[47:56])


# In[21]:


#Sample noise vectors from N(0,1) as latent variables and use the decoder to generate some images.
vector_10 = 3.5*np.random.randn(100,10)
plot_generator(vector_10, vae_10)


# In[22]:


#VAE with latent vector dimension=20, batch size=200
vae_20 = VAE(latent_dim = 20, lr = 0.003, batch_size = 200)
loss_20, out20 = vae_20.train(data_noise, data, 50)


# In[23]:


plt.plot(-np.array(loss_20)/784/200)
plt.title('latent dimension = 20 (batch_size = 200)')


# In[24]:


#show some examples reconstructed from the decoder.
plot_img(vae_20.reconstructor(data[47:56]), labels[47:56])


# In[25]:


#Sample noise vectors from N(0,1) as latent variables and use the decoder to generate some images.
vector_20 = 4*np.random.randn(100,20)
plot_generator(vector_20, vae_20)


# In[26]:


#VAE with latent vector dimension=20, batch size=100
vae_20_100 = VAE(latent_dim = 20, lr = 0.003, batch_size = 100)
loss_20_100, out20_100 = vae_20_100.train(data_noise, data, 50)


# In[27]:


plt.plot(-(np.array(loss_20_100))/784/300)
plt.title('latent dimension = 20 (batch_size = 100)')


# In[28]:


#show some examples reconstructed from the decoder.
plot_img(vae_20_100.reconstructor(data[47:56]), labels[47:56])


# In[29]:


#VAE with latent vector dimension=20, batch size=300
vae_20_300 = VAE(latent_dim = 20, lr = 0.003, batch_size = 300)
loss_20_300, out20_300 = vae_20_300.train(data_noise, data, 50)


# In[30]:


plt.plot(-(np.array(loss_20_300))/784/300)
plt.title('latent dimension = 20 (batch_size = 300)')


# In[40]:


#show some examples reconstructed from the decoder.
plot_img(vae_20_300.reconstructor(data[47:56]), labels[47:56])


# In[32]:


#show some examples reconstructed from the decoder.
plot_reconstructor(data[0:1024], vae_20_300)


# In[41]:


#show some examples reconstructed from the decoder.
plot_reconstructor(data[0:1024], vae_20)


# In[42]:


#show some examples reconstructed from the decoder.
plot_reconstructor(data[0:1024], vae_10)


# In[43]:


#show some examples reconstructed from the decoder.
plot_reconstructor(data[0:1024], vae_20_100)


# In[44]:


#show some examples reconstructed from the decoder.
plot_reconstructor(data[0:1024], vae_2)


# In[ ]:




