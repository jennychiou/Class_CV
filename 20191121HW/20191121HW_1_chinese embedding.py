#!/usr/bin/env python
# coding: utf-8

# In[9]:


import argparse
import logging
import os
import jieba
import math
import requests
import wiki as w
from gensim.models.fasttext import FastText
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


# In[10]:


from gensim.models import KeyedVectors
fasttext_300d_model = 'pre-trained/cc.zh.300/cc.zh.300.vec'
wordvectors_index = KeyedVectors.load_word2vec_format(fasttext_300d_model)


# In[11]:


wordvectors_index.most_similar('我')


# In[12]:


vec = wordvectors_index['我']
print(vec)
print('The Length of Vector:',len(vec))


# In[5]:


import pandas as pd
df = pd.read_csv('wikidata.csv')
df


# In[6]:


df['text'][0]


# In[7]:


import jieba
import numpy as np
jieba.set_dictionary('dict.txt.big')


# In[13]:


def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords  


# In[53]:


def seg_sentence(sentence):  
    sentence_seged = jieba.cut(sentence.strip())
    stopwords = stopwordslist('stops.txt')  # 加載停用詞的路徑  
    outstr = '' 
    for word in sentence_seged:  
        if word not in stopwords:  
            if word != '\t':  
                outstr += word  
                outstr += " "   #再次組合成【帶空格】的串
    return outstr


# In[55]:


inputs = open('input.txt', 'r', encoding = 'utf-8')
outputs = open('output.txt', 'w')

for line in inputs:  
    line_seg = seg_sentence(line)  # 這裏的返回值是字符串
    outputs.write(line_seg + '\n')

outputs.close()  
inputs.close()  


# In[91]:


f = open(r'output.txt')
text = []
for line in f:
    line = line.strip('\n').strip(' ')
    text.append(line)
print(text)
print(text[0].split(' '))


# In[96]:


import pandas as pd
final = []
for i in range(len(text)):
    x = text[i].split(' ')
    final.append(x)
    df_new = pd.DataFrame(final)
df_new


# In[102]:


df_new.iat[0,3]


# In[106]:


# 第一列元素個數
df_new.count(axis=1)[0]


# In[123]:


df_new.count(axis=1)[2]


# In[135]:


for a in range(0,3):
    for i in range(df_new.count(axis=1)[a]):
        getword = df_new.iat[a,i]
        print('Word:',getword)
        sim = wordvectors_index.most_similar(getword)
        print(sim,'\n')
        vec = wordvectors_index[getword]
        #print(vec)


# In[ ]:




