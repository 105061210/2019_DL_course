#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Before running this code, we should do data preprocessing by running ''hw4_2_105061210_pre.py

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pickle
import copy
from collections import Counter


# In[2]:


def load_preprocess():
    with open('preprocess.p', mode='rb') as in_file:
        return pickle.load(in_file)


# In[3]:


#load the preprocessed data (already from word to integer) and the tables for source and target mapping words to indices
(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = load_preprocess()


# In[4]:


class seq2seq(object):
    def __init__(self, rnn_size = 128, batch_size = 128, source_table = source_vocab_to_int, target_table = target_vocab_to_int, 
                lr = 0.001, num_layers = 3, embed_size_en = 200, embed_size_de = 200):
        self.rnn_size = rnn_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.embed_size_en = embed_size_en
        self.embed_size_de = embed_size_de
        self.lr = lr
        self.source_table = source_table
        self.target_table = target_table
        self.g = tf.Graph()
        with self.g.as_default():
            self.build()
            self.saver = tf.train.Saver()
            self.init_op = tf.global_variables_initializer()
        self.sess = tf.Session(graph = self.g)
    
    #padding every sentence in the batch to the max. sentence length in the batch
    def pad_sentence_batch(self, sentence_batch, pad_int):
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

    #creating batches for training and testing
    def batch_generator(self, sources, targets, batch_size, source_pad_int, target_pad_int):
        for batch_i in range(0, len(sources)//batch_size):
            start_i = batch_i * batch_size

            sources_batch = sources[start_i:start_i + batch_size]
            targets_batch = targets[start_i:start_i + batch_size]

            pad_sources_batch = np.array(self.pad_sentence_batch(sources_batch, source_pad_int))
            pad_targets_batch = np.array(self.pad_sentence_batch(targets_batch, target_pad_int))

            pad_targets_lengths = []
            for target in pad_targets_batch:
                pad_targets_lengths.append(len(target))

            pad_source_lengths = []
            for source in pad_sources_batch:
                pad_source_lengths.append(len(source))

            yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths
    
    #encoder (including the embedding layer) using LSTM rnn cells
    def encoding_layer(self, rnn_inputs, rnn_size, num_layers, keep_prob, source_vocab_size, encoding_embedding_size):
        embed = tf.contrib.layers.embed_sequence(rnn_inputs, vocab_size=source_vocab_size, embed_dim=encoding_embedding_size)
        stacked_cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size), keep_prob) for _ in range(num_layers)])
        outputs, state = tf.nn.dynamic_rnn(stacked_cells, embed, dtype=tf.float32)
        return outputs, state
    
    #add special token <GO>  in front of all target data sequences
    def process_decoder_input(self, target_data, target_vocab_to_int, batch_size):
        go_id = target_vocab_to_int['<GO>']
        after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
        after_concat = tf.concat( [tf.fill([batch_size, 1], go_id), after_slice], 1)
        return after_concat    
    
    #using training helper (for training)
    def decoding_layer_train(self, encoder_state, dec_cell, dec_embed_input, target_sequence_length, max_summary_length, output_layer, keep_prob):
        dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=keep_prob)
        helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, target_sequence_length)
        decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, encoder_state, output_layer)
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=max_summary_length)
        return outputs
    
    #using GreedyEmbedding helper (for later inference)
    def decoding_layer_infer(self, encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id, max_target_sequence_length, vocab_size, output_layer, batch_size, keep_prob):
        dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=keep_prob)
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, tf.fill([batch_size], start_of_sequence_id), end_of_sequence_id)
        decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, encoder_state, output_layer)
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True, maximum_iterations=max_target_sequence_length)
        return outputs
    
    #decoder
    def decoding_layer(self, dec_input, encoder_state, target_sequence_length, max_target_sequence_length, rnn_size, num_layers, target_vocab_to_int, target_vocab_size, batch_size, keep_prob, decoding_embedding_size):
        target_vocab_size = len(target_vocab_to_int)
        dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
        dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
        cells = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(num_layers)])
    
        with tf.variable_scope("decode"):
            output_layer = tf.layers.Dense(target_vocab_size)
            train_output = self.decoding_layer_train(encoder_state, 
                                            cells, 
                                            dec_embed_input, 
                                            target_sequence_length, 
                                            max_target_sequence_length, 
                                            output_layer, 
                                            keep_prob)

        with tf.variable_scope("decode", reuse=True):
            infer_output = self.decoding_layer_infer(encoder_state, 
                                            cells, 
                                            dec_embeddings, 
                                            target_vocab_to_int['<GO>'], 
                                            target_vocab_to_int['<EOS>'], 
                                            max_target_sequence_length, 
                                            target_vocab_size, 
                                            output_layer,
                                            batch_size,
                                            keep_prob)

        return train_output, infer_output
    
    def build(self):
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        targets = tf.placeholder(tf.int32, [None, None], name='targets') 
        target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
        max_target_len = tf.reduce_max(target_sequence_length)
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        
        en_outputs, en_states = self.encoding_layer(tf.reverse(inputs, [-1]), 
                                             self.rnn_size, 
                                             self.num_layers, 
                                             keep_prob, 
                                             len(self.source_table), 
                                             self.embed_size_en)
    
        dec_input = self.process_decoder_input(targets, 
                                          self.target_table, 
                                          self.batch_size)
    
        train_output, infer_output = self.decoding_layer(dec_input,
                                                   en_states, 
                                                   target_sequence_length, 
                                                   max_target_len,
                                                   self.rnn_size,
                                                   self.num_layers,
                                                   self.target_table,
                                                   len(self.target_table),
                                                   self.batch_size,
                                                   keep_prob,
                                                   self.embed_size_de)

        self.training_logits = tf.identity(train_output.rnn_output, name='logits')
        self.inference_logits = tf.identity(infer_output.sample_id, name='predictions')
        masks = tf.sequence_mask(target_sequence_length, max_target_len, dtype=tf.float32, name='masks')
        
        #cost using 'Weighted cross-entropy loss for a sequence of logits'.
        self.cost = tf.contrib.seq2seq.sequence_loss(self.training_logits, targets, masks, name = 'cost')
        optimizer = tf.train.AdamOptimizer(self.lr)
        #doing gradient clipping
        gradients = optimizer.compute_gradients(self.cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients, name = 'train_op')
    
    #padding(so the target and logit have the same length) and then returns the average of where the target and logits have the same value
    def get_accuracy(self, target, logits):
        max_seq = max(target.shape[1], logits.shape[1])
        if max_seq - target.shape[1]:
            target = np.pad(target, [(0,0),(0,max_seq - target.shape[1])], 'constant')
        if max_seq - logits.shape[1]:
            logits = np.pad(logits, [(0,0),(0,max_seq - logits.shape[1])], 'constant')
        return np.mean(np.equal(target, logits))
    
    def train(self, source_int_text, target_int_text, epochs=13):
        train_source = source_int_text[self.batch_size:]
        train_target = target_int_text[self.batch_size:]
        valid_source = source_int_text[:self.batch_size]
        valid_target = target_int_text[:self.batch_size]
        
        (valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths) = next(self.batch_generator(valid_source,
                                                                                                                     valid_target,
                                                                                                                     self.batch_size,
                                                                                                                     self.source_table['<PAD>'],
                                                                                                                     self.target_table['<PAD>']))
        
        self.sess.run(self.init_op)
        train_loss = []
        valid_loss = []
        train_accuracy = []
        valid_accuracy = []
        for epoch_i in range(epochs):
            train_batch_loss = []
            valid_batch_loss = []
            for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(self.batch_generator(train_source, train_target, self.batch_size, source_vocab_to_int['<PAD>'], target_vocab_to_int['<PAD>'])):
                _, loss = self.sess.run(['train_op', self.cost], feed_dict={'inputs:0': source_batch, 'targets:0': target_batch,'target_sequence_length:0': targets_lengths,'keep_prob:0': 0.5})
                val_loss = self.sess.run(self.cost, feed_dict={'inputs:0': valid_sources_batch, 'targets:0': valid_targets_batch, 'target_sequence_length:0': valid_targets_lengths,'keep_prob:0': 1.0})
                train_batch_loss.append(loss)
                valid_batch_loss.append(val_loss)
                
                if batch_i % 500 == 0 and batch_i > 0:
                    batch_train_logits = self.sess.run(self.inference_logits, feed_dict={'inputs:0': source_batch, 'target_sequence_length:0': targets_lengths, 'keep_prob:0': 1.0})
                    batch_valid_logits = self.sess.run(self.inference_logits, feed_dict={'inputs:0': valid_sources_batch,'target_sequence_length:0': valid_targets_lengths, 'keep_prob:0': 1.0})

                    train_acc = self.get_accuracy(target_batch, batch_train_logits)
                    valid_acc = self.get_accuracy(valid_targets_batch, batch_valid_logits)

                    print('Epoch {:>3}    Batch {:>4}/{} \n Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                      .format(epoch_i, batch_i, len(source_int_text) // self.batch_size, train_acc, valid_acc, loss))
        
            train_loss.append(np.mean(train_batch_loss))
            valid_loss.append(np.mean(valid_batch_loss))
            train_accuracy.append(train_acc)
            valid_accuracy.append(valid_acc)
        
        self.saver.save(self.sess, 'checkpoints/model')
        #save_params('checkpoints/param')
        print('Model Trained and Saved')
        return train_loss, valid_loss, train_accuracy, valid_accuracy


# In[5]:


#construct a sequence to sequence model
model = seq2seq()


# In[ ]:


train_loss, valid_loss, train_accuracy, valid_accuracy = model.train(source_int_text, target_int_text, epochs = 10)


# In[5]:


def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()
    return data

def sentence_to_sequence(sentence, vocab_to_int):
    results = []
    for word in sentence.split(" "):
        if word in vocab_to_int:
            results.append(vocab_to_int[word])
        else:
            results.append(vocab_to_int['<UNK>'])            
    return results


# In[6]:


translate_text = load_data('test.txt')
translate_sentences = translate_text.split('\n')
batch_size = 128
print(len(translate_sentences))

_, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = load_preprocess()

if translate_text[0] == '\ufeff':
    translate_sentences[0]=translate_sentences[0].split('\ufeff')[1]
    
fr = []
for i in range(len(translate_sentences)-1):
    translate_sentence = translate_sentences[i]
    translate_sentence = sentence_to_sequence(translate_sentence, source_vocab_to_int)

    g = tf.Graph()
    with tf.Session(graph=g) as sess:
    
        loader = tf.train.import_meta_graph('checkpoints/model' + '.meta')
        loader.restore(sess, 'checkpoints/model')

        inputs = g.get_tensor_by_name('inputs:0')
        logits = g.get_tensor_by_name('predictions:0')
        target_sequence_length = g.get_tensor_by_name('target_sequence_length:0')
        keep_prob = g.get_tensor_by_name('keep_prob:0')

        translate_logits = sess.run(logits, feed_dict={inputs: [translate_sentence]*batch_size,
                                         target_sequence_length: [len(translate_sentence)*2]*batch_size,
                                         keep_prob: 1.0})[0]
    string = ' '
    fr_list = [target_int_to_vocab[i] for i in translate_logits]
    fr_list = fr_list[0:len(fr_list)-1]
    fr.append(string.join(fr_list))
    print()
    print('Source (English)')
    print('  Word Indices:      {}'.format([i for i in translate_sentence]))
    print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))
    print()
    print('Translation (French)')
    print('  Word Indices:      {}'.format([i for i in translate_logits]))
    print('  French Words: {}'.format(" ".join([target_int_to_vocab[i] for i in translate_logits])))
    print('--------------------------------------------------------------------------------------')


# In[9]:


string = '\n'
fr=string.join(fr)
print(fr)


# In[ ]:


output_file = os.path.join('test_105061210.txt')
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(fr)


# In[ ]:

import matplotlib.pyplot as plt
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


plot_loss('seq2seq', train_loss, valid_loss)


# In[ ]:


plot_acc('seq2seq', train_accuracy, valid_accuracy)


# In[ ]:




