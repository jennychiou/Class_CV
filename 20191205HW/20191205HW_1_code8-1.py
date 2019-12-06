#!/usr/bin/env python
# coding: utf-8

# In[5]:


import keras
import numpy as np
from keras import layers
import random
import sys


# In[2]:


def reweight_distribution(original_distribution, temperature=2):
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)


# In[3]:


ori_dstri = np.array([0.8, 0.1, 0.1])   # a、b、c 的機率分布

new_dstri = reweight_distribution(ori_dstri, temperature=0.01)  # 使用溫度 0.01 
print(new_dstri)    # [1.00000000e+00 4.90909347e-91 4.90909347e-91]

new_dstri = reweight_distribution(ori_dstri, temperature=2) # 使用預設溫度 2
print(new_dstri)    # [0.58578644 0.20710678 0.20710678]

new_dstri = reweight_distribution(ori_dstri, temperature=10) # 使用溫度 10
print(new_dstri)    # [0.38102426 0.30948787 0.30948787]


# In[6]:


path = keras.utils.get_file(        # 取得文本檔案
    'nietzsche.txt',                # 檔名 
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')  # 檔案位置
text = open(path, encoding='utf-8').read().lower()    # 讀取文本內容，並轉成小寫
print('Corpus length:', len(text))  # 文本長度為 600893


# In[7]:


maxlen = 60     # 每次 (step) 從文本中萃取 60 個字元作為序列資料
step = 3        # 每 3 個字元為一個 step 進行萃取 

sentences = []  # 存放萃取出的序列資料
next_chars = [] # 存放對應目標 (萃取出來的序列資料的後一個字元)

# 萃取序列資料 
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('Number of sequences:', len(sentences))   # 共萃取出 200278 個序列資料


# In[8]:


# 產生文本中的 "唯一" 字元串列 (文本轉成 set 將重複字元刪除)
chars = sorted(list(set(text))) 
print('Unique characters:', len(chars)) # 文本共使用 57 種字元


# In[9]:


# 將各個字元對應到 "chars" 串列中的索引值成為字典 (dict) 格式。即 {'\n': 0,' ': 1, '!': 2,…}
char_indices = dict((char, chars.index(char)) for char in chars)

# 將字元經 One-hot 編碼成二元陣列
print('Vectorization...')


# In[10]:


x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
print(x.shape)  # (200278, 60, 57)
print(y.shape)  # (200278, 57)


# In[11]:


model = keras.models.Sequential()   # 建立序列式模型
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))


# In[12]:


optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# In[13]:


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature     # 重新加權計算 (熵)
    exp_preds = np.exp(preds)               
    preds = exp_preds / np.sum(exp_preds)   
    probas = np.random.multinomial(1, preds, 1) # 丟入多項式分布中
    return np.argmax(probas)


# In[14]:


for epoch in range(1, 60):  # 共 60 個訓練週期 (次數)
    print('epoch', epoch)
    model.fit(x, y,         # 用萃取出來的 x, y 開始進行訓練
              batch_size=128,
              epochs=1)
    # 隨機選擇文本中的某段 60 個字元
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index: start_index + maxlen]
    print('--- 隨機初始文字: "' + generated_text + '"')

    # 嘗試使用一系列不同 temperature 生成文字
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperature:', temperature)
        sys.stdout.write(generated_text)

        # 每個 temperature 生成 400 個字元
        for i in range(400):    
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.
            preds = model.predict(sampled, verbose=0)[0]    # 產生字元機率分布
            next_index = sample(preds, temperature) # 重新加權並取樣，回傳字元索引
            next_char = chars[next_index]           # 確認新字元
            generated_text += next_char             # 新字元加到文字的後方
            generated_text = generated_text[1:]     # 重新取得含括新字元的文字繼續生成下一個字元

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


# In[ ]:




