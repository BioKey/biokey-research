
# coding: utf-8

# In[1]:


import sys
import os
sys.path.append("..")
from biokey.data import DataInterface
import biokey.tools
import credentials


# In[2]:


import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt


# In[3]:


data = DataInterface(credentials.postgres)


# In[4]:


datasets = data.get_user_sets('1d63b44d-a7cb-4ee6-b228-b0ff5b7d086f')
datasets.train = biokey.tools.filter_keys(datasets.train)
datasets.test = biokey.tools.filter_keys(datasets.test)


# # Form Dwell Sequence Frames

# In[5]:


def generate_arrays(df):
    downsample_rate = 10
    train = df.copy()[['key_enum', 'down', 'up', 'is_user']]
    train.down = ((df.down - df.down.min())/downsample_rate).round().astype('int64')
    train.up = ((df.up - df.down.min())/downsample_rate).round().astype('int64')
    train.is_user = (train.is_user).astype('int32')
    train['not_user'] = (train.is_user == False).astype('int32')
    
    sample_freq = 100*60
    sample_length = train.up.max()
    # only select first n frames
    n = 100
    sample_length = int(sample_length/sample_freq)*n
    num_rows = int(sample_length/sample_freq)
    num_keys = train.key_enum.unique().max() + 1

    x_series = np.zeros((num_rows*sample_freq , num_keys), dtype='int16')
    y_series = np.zeros((num_rows*sample_freq, 3), dtype='int16')
    y_series[:,2] = 1 # Set inactive to true
    #(is_user, is_imposter, inactive)
    for x in train.itertuples():
        x_series[x.down:x.up, x.key_enum] = 1
        y_series[x.down:x.up, 0] = x.is_user
        y_series[x.down:x.up, 1] = x.not_user
        y_series[x.down:x.up, 2] = 0
    x_batches = np.array(np.array_split(x_series, num_rows))
    y_batches = np.array(np.array_split(y_series, num_rows))
    y_sample = y_batches.sum(axis=1)
    y_sample[:,0] = (y_sample[:,0] > y_sample[:,1]).astype('int16')
    y_sample[:,1] = (y_sample[:,0] < y_sample[:,1]).astype('int16')
    y_sample[:,2] = ((y_sample[:,0] == 0) & (y_sample[:,1] == 0)).astype('int16')
    
    return x_batches, y_sample


# In[6]:


X, Y = generate_arrays(datasets.train)


# In[7]:


X_test, Y_test = generate_arrays(datasets.test)


# # Use Keras

# In[8]:


from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import TensorBoard
from sklearn.metrics import roc_curve, auc
from phased_lstm_keras.PhasedLSTM import PhasedLSTM as PLSTM


# In[9]:


# Parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 100

# Network Parameters
input_shape = (X.shape[1], X.shape[2])
n_hidden_1 = 32 # 1st layer number of features
n_hidden_2 = 32 # 2nd layer number of features
n_classes = Y.shape[1] # Number of classes to predict


# # PhasedLSTM

# In[ ]:

'''
n_hidden_1 = 32
# create model
model_PLSTM = Sequential()
model_PLSTM.add(PLSTM(n_hidden_1, input_shape=input_shape, implementation=2))
model_PLSTM.add(Dense(n_classes, activation='softmax'))
# Compile model
model_PLSTM.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model_PLSTM.summary()
tbCallBack = TensorBoard(log_dir='./Graph/plstm'+str(n_hidden_1), histogram_freq=0, write_graph=True, write_images=True)


# In[ ]:


model_PLSTM.fit(X, Y, epochs=training_epochs, batch_size=batch_size, callbacks=[tbCallBack])
model_PLSTM.save('model_plstm'+str(n_hidden_1)+'.h5')
'''

# In[ ]:


n_hidden_1 = 256
# create model
model_PLSTM = Sequential()
model_PLSTM.add(PLSTM(n_hidden_1, input_shape=input_shape, implementation=2))
model_PLSTM.add(Dense(n_classes, activation='softmax'))
# Compile model
model_PLSTM.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model_PLSTM.summary()
tbCallBack = TensorBoard(log_dir='./Graph/plstm'+str(n_hidden_1)+'_e'+str(training_epochs), histogram_freq=0, write_graph=True, write_images=True)


# In[ ]:


model_PLSTM.fit(X, Y, epochs=training_epochs, batch_size=batch_size, callbacks=[tbCallBack])
model_PLSTM.save('model_plstm'+str(n_hidden_1)+'.h5')

'''
# # LSTM

# In[ ]:


n_hidden_1 = 32
# create model
model_LSTM = Sequential()
model_LSTM.add(PLSTM(n_hidden_1, input_shape=input_shape, implementation=2))
model_LSTM.add(Dense(n_classes, activation='softmax'))
# Compile model
model_LSTM.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model_LSTM.summary()
tbCallBack = TensorBoard(log_dir='./Graph/lstm'+str(n_hidden_1), histogram_freq=0, write_graph=True, write_images=True)


# In[ ]:


model_LSTM.fit(X, Y, epochs=training_epochs, batch_size=batch_size, callbacks=[tbCallBack])
model_PLSTM.save('model_lstm'+str(n_hidden_1)+'.h5')


# In[ ]:


n_hidden_1 = 256
# create model
model_LSTM = Sequential()
model_LSTM.add(PLSTM(n_hidden_1, input_shape=input_shape, implementation=2))
model_LSTM.add(Dense(n_classes, activation='softmax'))
# Compile model
model_LSTM.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model_LSTM.summary()
tbCallBack = TensorBoard(log_dir='./Graph/lstm'+str(n_hidden_1), histogram_freq=0, write_graph=True, write_images=True)


# In[ ]:


model_LSTM.fit(X, Y, epochs=training_epochs, batch_size=batch_size, callbacks=[tbCallBack])
model_PLSTM.save('model_lstm'+str(n_hidden_1)+'.h5')
'''

# # Assess Results

# In[ ]:


def generate_results(y_test, y_score):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.show()
    print('AUC: %f' % roc_auc)

