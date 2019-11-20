#!/usr/bin/env python
# coding: utf-8

# # 單字的 one-hot encoding

# In[1]:


import numpy as np

# 初始資料：每一個樣本是一個輸入項目(在這範例中，樣本是一個句子，但也可以是整個文件)
samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# 建立資料中所有 tokens 的索引
token_index = {}

for sample in samples:
    for word in sample.split():  # 分詞並移除樣本中的標點符號與特殊字元
        if word not in token_index:
            token_index[word] = len(token_index) + 1  # 為每個文字指定一個唯一索引

max_length = 10  # 將樣本向量化。每次只專注處理每個樣本中的第一個 max_length 文字

results = np.zeros(shape=(len(samples),  # 用來儲存結果的 Numpy array
                          max_length,
                          max(token_index.values()) + 1))  
print('shape :',results.shape)


# In[2]:


for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.
print('token_index :',token_index)


# # 字元的 one-hot encoding

# In[3]:


import string
import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
characters = string.printable  # 所有可印出的 ASCII 字元的字串, '0123456789abc....'
print('字元長度',len(characters))

token_index = dict(zip(characters, range(1, len(characters) + 1)))

max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
print('shape :',results.shape) 


# In[4]:


for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        results[i, j, index] = 1.
print(results[0][0])


# # 用 Keras 做文字的 one-hot encoding

# In[5]:


from keras.preprocessing.text import Tokenizer

# 初始資料
samples = ['The cat sat on the mat.', 'The dog ate my homework.']

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples) 

sequences = tokenizer.texts_to_sequences(samples)
print('sequences :',sequences)

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
print('one_hot shape :',one_hot_results.shape)

word_index = tokenizer.word_index
print('word_index :',word_index)
print('找到 %s 個唯一的 tokens.' % len(word_index)) 


# # 使用雜湊技巧的單字 one-hot encoding

# In[6]:


samples = ['The cat sat on the mat.', 'The dog ate my homework.']

dimensionality = 1000
max_length = 10

results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1.
print('shape :',results.shape)


# # 使用嵌入向量層學習文字嵌入向量

# In[7]:


# 建立一個嵌入層 (Embedding Layer)
from keras.layers import Embedding

# 建立嵌入向量層至少須指定兩個參數
embedding_layer = Embedding(1000, 64)


# In[8]:


# 載入 IMDB, 整理成適合供 Embedding 層使用的資料
from keras.datasets import imdb
from keras import preprocessing

# 設定作為特徵的文字數量
max_features = 10000
# 在 20 個文字之後切掉文字資料
maxlen = 20


# In[11]:


#(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# restore np.load for future normal usage
np.load = np_load_old
print('x_train shape :',x_train.shape)

# 將資料以整數 lists 載入
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
print(x_train.shape)
print(x_train[0])

# 將整數 lists 轉換為 2D 整數張量
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)


# In[12]:


# 把 IMDB 資料提供給 Embedding layer和分類器
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen)) # ←1...

model.add(Flatten()) # ← 2...

model.add(Dense(1, activation='sigmoid')) # ← 在頂部加上分類器
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, 
                    y_train,epochs=10,
                    batch_size=32,
                    validation_split=0.2)


# In[ ]:




