
# coding: utf-8

# # The Office Neural Network
# 
# ### The goal of this notebook is to create a Recurrent Neural Net in order to generate new scripts of The Office
# 
# ##### The first step is going to be condensing this data frame down into a .txt file

# In[153]:


import pandas as pd
import numpy as np
import os
import pickle
import helpers

import warnings
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard

# In[136]:


df = pd.read_csv('cleaned-data.csv')


# In[137]:


df.head(10)


# Here I create a helper function that given a dataframe will condense it into a string

# In[138]:


file = ""
for i in range(len(df['id'])):
    file += str(df['speaker'].iloc[i] + ': ' + str(df['line_text'].iloc[i])) + '\n'


# In[139]:


print(file[0:500])


# now we write this text data to a file so we can reopen it earlier without having to rerun code!

# In[140]:


script = open('./data/script.txt', 'w')
script.write(str(file.encode("utf-8")))
script.close()


# In[141]:


from collections import Counter

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    
    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return (vocab_to_int, int_to_vocab)


# In[142]:


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    tokenized_text = {
        '.':'<PERIOD>',
        ',':'<COMMA>',
        '"':'<QUOTATION_MARK>',
        ';':'<SEMICOLON>',
        '!':'<EXCLAMATION_MARK>',
        '?':'<QUESTION_MARK>',
        '(':'<LEFT_PAREN>',
        ')':'<RIGHT_PAREN>',
        '--':'<DASH>',
        '\n':'<RETURN>'
    }
    
    return tokenized_text


# In[143]:


helpers.preprocess_and_save_data('./data/script.txt', token_lookup, create_lookup_tables)


# In[146]:


int_text, vocab_to_int, int_to_vocab, token_dict = helpers.load_preprocess()


# In[155]:


def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    input = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    return(input, targets, learning_rate)


# ## RNN TIME!!!

# In[156]:


def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
 
    lstm_layers = 2
    
    # Basic LSTM cell
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([lstm] * lstm_layers)
    
    # Getting an initial state of all zeros
    initial_state = tf.identity(cell.zero_state(batch_size, tf.float32), name='initial_state')
    
    return(cell, initial_state)


# In[157]:


def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    embedding_weight = tf.Variable(tf.truncated_normal((vocab_size, embed_dim), stddev = 0.01))
    embed_layer = tf.nn.embedding_lookup(embedding_weight, input_data)

    return embed_layer


# In[158]:


def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype = tf.float32)
    final_state = tf.identity(final_state, name='final_state')
    
    return outputs, final_state


# In[159]:


def build_nn(cell, rnn_size, input_data, vocab_size):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :return: Tuple (Logits, FinalState)
    """
    embed = get_embed(input_data, vocab_size, rnn_size)
    outputs, final_state = build_rnn(cell, embed)
    logits = tf.contrib.layers.fully_connected(outputs, 
                                               vocab_size, 
                                               weights_initializer = tf.truncated_normal_initializer(stddev = 0.01), 
                                               activation_fn=None)
    
    
    return logits, final_state


def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    num_items = len(int_text)
    num_batches = num_items // (batch_size * seq_length)
    batches = np.zeros(shape=(num_batches, 2, batch_size, seq_length), dtype=np.int32)
    for item in range(num_batches):
        for batch in range(batch_size):
            start_x = batch * batch_size * seq_length + item * seq_length
            end_x = start_x + seq_length
            if end_x < num_items:
                start_y = start_x + 1
                end_y = start_y + seq_length
                batches[item, 0, batch, :] = int_text[start_x:end_x]
                batches[item, 1, batch, :] = int_text[start_y:end_y]

    return batches


# In[161]:


# Number of Epochs
num_epochs = 125
# Batch Size
batch_size = 512
# RNN Size
rnn_size = 1024
# Sequence Length
seq_length = 30
# Learning Rate
learning_rate = 0.01
# Show stats for every n number of batches
show_every_n_batches = 100

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
save_dir = './save'


# In[162]:


from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
    train_op = optimizer.apply_gradients(capped_gradients)


# In[ ]:


batches = get_batches(int_text, batch_size, seq_length)
now = datetime.now()
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)
            NAME = "epoch-{}-batch-{}-{}".format(epoch_i, batch_i, now.strftime("%Y%m%d-%H%M%S"))
            tensorboard = TensorBoard(log_dir='logs/main/{}'.format(NAME))

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')


# In[ ]:


helpers.save_params((seq_length, save_dir))


# In[ ]:


_, vocab_to_int, int_to_vocab, token_dict = helpers.load_preprocess()
seq_length, load_dir = helpers.load_params()


# In[ ]:




def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    InputTensor = loaded_graph.get_tensor_by_name("input:0")
    InitialStateTensor = loaded_graph.get_tensor_by_name("initial_state:0")
    FinalStateTensor = loaded_graph.get_tensor_by_name("final_state:0")
    ProbsTensor = loaded_graph.get_tensor_by_name("probs:0")
    
    return InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor


# In[ ]:




def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    word_id = np.random.choice(len(probabilities), p=probabilities)
    return int_to_vocab[word_id]


# In[ ]:


gen_length = 500
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'michael'

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word + ':']
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})

        pred_word = pick_word(probabilities[0][dyn_seq_length - 1], int_to_vocab)
        #pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)
    
    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')
    list_tv_script = tv_script.split('\\n')
    print(tv_script)

pred_script = open('./data/new_script_08_31_2018.txt', 'w')
pred_script.write(tv_script)
pred_script.close()
