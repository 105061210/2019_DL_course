#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# In[2]:


imdb = tf.keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3





train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = tf.keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)


# In[3]:


class RNN(object):
    def __init__(self, n_words = 10000, seq_len = 256, cell_type = 'vanilla', hidden_size = 128, num_layers = 1, batch_size = 125, lr = 0.00075, embed_size = 256):
        self.n_words = n_words
        self.seq_len = seq_len
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lr = lr
        self.embed_size = embed_size
        self.g = tf.Graph()
        with self.g.as_default():
            self.build()
            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()
        self.sess = tf.Session(graph = self.g)
        
    def batch_generator(self, X, y = None, batch_size = 64):
        n_batches = len(X)//batch_size
        X = X[:n_batches*batch_size]
        if y is not None:
            y = y[:n_batches*batch_size]
        for i in range(0, len(X), batch_size):
            if y is not None:
                yield X[i: i+batch_size], y[i: i+batch_size]
            else:
                yield X[i: i+batch_size]
                
    def build(self):
        #placeholders for inputs
        tf_x = tf.placeholder(tf.int32, shape = [None, self.seq_len], name = 'tf_x')
        tf_y = tf.placeholder(tf.float32, shape = [None], name = 'tf_y')
        keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
            
        #embedding layer
        self.embedding = tf.Variable(tf.random_uniform([self.n_words, self.embed_size], -1.0, 1.0), name='embedding')
        self.embedded_x = tf.nn.embedding_lookup(self.embedding, tf_x, name = 'embedded_x')
            
        #which type of RNN cell
        if self.cell_type == 'vanilla':
            cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_size)
        elif self.cell_type == 'GRU':
            cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        elif self.cell_type == 'LSTM':
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
        else:
            print('unknown cell type!')
            
        #stack RNN cell together
        cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = keep_prob) for i in range(self.num_layers)])
            
        #define initial state
        self.initial_state = cells.zero_state(self.batch_size, tf.float32)
        print('\n initial state : ', self.initial_state)
            
        #create RNN using cells and their states
        outputs, self.final_state = tf.nn.dynamic_rnn(cells, self.embedded_x, initial_state = self.initial_state)
        print('\n outputs : ', outputs)
        print('\n final_state : ', self.final_state)
            
        #adding FC layer on top of RNN output
        logits = tf.layers.dense(inputs = outputs[:, -1], units = 1, activation = None, name = 'logits')
        logits_sq = tf.squeeze(logits, name = 'logits_sq')
        print('\n logits : ', logits_sq)
            
        #predictions(probabilities and labels)
        prob = tf.nn.sigmoid(logits_sq, name = 'prob')
        predictions = {
            'prob': prob,
            'labels': tf.cast(tf.round(prob), tf.float32, name = 'labels')
        }
        print('\n predictions : ', predictions)
        
        correct_pred = tf.equal(predictions['labels'], tf_y, name = 'correct_pred')
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name = 'accuracy')
        
        #cost function(cross entropy) and optimizer
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf_y, logits = logits_sq), name = 'cost')
        optimizer = tf.train.AdamOptimizer(self.lr)
        train_op = optimizer.minimize(cost, name = 'train_op')
    
    def train(self, X_train, y_train, X_val, y_val, epochs):
        #with tf.Session(graph = self.g) as self.sess:
        self.sess.run(self.init_op)
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        for epoch in range(1, epochs+1):
            state = self.sess.run(self.initial_state)
            batch_loss = []
            batch_acc = []
            for batch_X, batch_y in self.batch_generator(X_train, y_train, self.batch_size):
                _, loss, acc = self.sess.run(['train_op', 'cost:0', 'accuracy:0'], feed_dict={'tf_x:0': batch_X, 'tf_y:0': batch_y, 'keep_prob:0': 0.5, self.initial_state: state})
                batch_loss.append(loss)
                batch_acc.append(acc)
            train_loss.append(np.mean(batch_loss))
            print('epoch: ', epoch, '    train loss: ', train_loss[epoch-1])
            train_acc.append(np.mean(batch_acc))
            print('epoch: ', epoch, '    train accuracy: ', train_acc[epoch-1])
                
            val_batch_loss = []
            val_batch_acc = []
            for batch_X, batch_y in self.batch_generator(X_val, y_val, self.batch_size):
                v_loss, v_acc = self.sess.run(['cost:0', 'accuracy:0'], feed_dict={'tf_x:0': batch_X, 'tf_y:0': batch_y, 'keep_prob:0': 1.0, self.initial_state: state})
                val_batch_loss.append(v_loss)
                val_batch_acc.append(v_acc)
            val_loss.append(np.mean(val_batch_loss))
            print('epoch: ', epoch, '    test loss: ', val_loss[epoch-1])
            val_acc.append(np.mean(val_batch_acc))
            print('epoch: ', epoch, '    test accuracy: ', val_acc[epoch-1])
            print('--------------------------------------------------------')
            if epoch %5 == 0:
                self.saver.save(self.sess, 'model/%d.ckpt' %epoch)
        return train_loss, train_acc, val_loss, val_acc
    
    def predict(self, X, label = True):
        preds = []
        #with tf.Session(graph = self.g) as sess:
            #self.saver.restore(self.sess, tf.train.latest_checkpoint('model/'))
        state = self.sess.run(self.initial_state)
        for batch_X in self.batch_generator(X, None, self.batch_size):
            feed_dict={'tf_x:0': batch_X, 'keep_prob:0': 1.0, self.initial_state: state}
            if not label:
                pred = self.sess.run('prob:0', feed_dict = feed_dict)
            else:
                pred = self.sess.run('Round:0', feed_dict = feed_dict)
                
            preds.append(pred)   
        return np.concatenate(preds)
    


# In[4]:


def plot_loss(rnn_type, train_loss, test_loss):
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label = rnn_type + '  training loss')
    plt.plot(test_loss, label = rnn_type + '  test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(rnn_type + " RNN Model Loss")
    plt.legend(loc='best')
    plt.show()
    
def plot_acc(rnn_type, train_acc, test_acc):
    plt.figure(figsize=(8, 6))
    plt.plot(train_acc, label = rnn_type + '  training acc')
    plt.plot(test_acc, label = rnn_type + '  test acc')
    plt.xlabel('epoch')
    plt.ylim([0.0, 1.05])
    plt.ylabel('accuracy')
    plt.title(rnn_type + " RNN Model Accuracy")
    plt.legend(loc='best')
    plt.show()


# In[ ]:


from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

def plot_roc(rnn_type, test_labels, y_pred):
    fpr,tpr, _ = roc_curve(test_labels, y_pred)
    roc_auc = auc(fpr,tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(rnn_type + ' Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
def plot_prc(rnn_type, test_labels, y_pred):
    average_precision = average_precision_score(test_labels, y_pred)
    precision, recall, _ = precision_recall_curve(test_labels, y_pred)
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(rnn_type + ' Precision-Recall curve: AP={0:0.2f}'.format(average_precision))


# In[ ]:


vanilla_rnn = RNN(hidden_size = 128, num_layers = 1, batch_size = 125, lr = 0.00075, embed_size = 256)


# In[ ]:


train_loss, train_acc, test_loss, test_acc = vanilla_rnn.train(train_data, train_labels, test_data, test_labels, epochs = 20)


# In[ ]:


plot_loss('vanilla', train_loss, test_loss)


# In[ ]:


plot_acc('vanilla', train_acc, test_acc)


# In[ ]:


vanilla_y_pred = vanilla_rnn.predict(test_data, label = True)


# In[ ]:


plot_roc('vanilla', test_labels, vanilla_y_pred)


# In[ ]:


plot_prc('vanilla', test_labels, vanilla_y_pred)


# In[ ]:


gru_rnn = RNN( cell_type = 'GRU', hidden_size = 128, num_layers = 1, batch_size = 125, lr = 0.00075, embed_size = 256)


# In[ ]:





# In[ ]:


gru_train_loss, gru_train_acc, gru_test_loss, gru_test_acc = gru_rnn.train(train_data, train_labels, test_data, test_labels, epochs = 30)


# In[ ]:


plot_loss('GRU', gru_train_loss, gru_test_loss)


# In[ ]:


plot_acc('GRU', gru_train_acc, gru_test_acc)


# In[ ]:


gru_y_pred = gru_rnn.predict(test_data, label = True)


# In[ ]:


plot_roc('GRU', test_labels, gru_y_pred)


# In[ ]:


plot_prc('GRU', test_labels, gru_y_pred)


# In[ ]:





# In[ ]:


lstm_rnn = RNN( cell_type = 'LSTM', hidden_size = 128, num_layers = 1, batch_size = 125, lr = 0.00075, embed_size = 256)


# In[ ]:


lstm_train_loss, lstm_train_acc, lstm_test_loss, lstm_test_acc = lstm_rnn.train(train_data, train_labels, test_data, test_labels, epochs = 30)


# In[ ]:


plot_loss('LSTM', lstm_train_loss, lstm_test_loss)


# In[ ]:


plot_acc('LSTM', lstm_train_acc, lstm_test_acc)


# In[ ]:


lstm_y_pred = lstm_rnn.predict(test_data, label = True)


# In[ ]:


plot_roc('LSTM', test_labels, lstm_y_pred)


# In[ ]:


plot_prc('LSTM', test_labels, lstm_y_pred)


# In[ ]:





# In[ ]:




