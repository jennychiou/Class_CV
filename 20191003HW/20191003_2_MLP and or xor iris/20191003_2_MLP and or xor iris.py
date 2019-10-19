
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


model = Sequential()
model.add(Dense(8, input_dim=2))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))


# In[3]:


# XOR
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
sgd = SGD(lr = 0.1)
model.compile(loss = 'binary_crossentropy',optimizer = 'sgd',metrics = ['accuracy'])
model.fit(x, y, epochs = 1000, batch_size = 1)
print(model.predict_proba(x))


# In[4]:


# OR
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[1]])
sgd = SGD(lr = 0.1)
model.compile(loss = 'binary_crossentropy',optimizer = 'sgd',metrics = ['accuracy'])
model.fit(x, y, epochs = 1000, batch_size = 1)
print(model.predict_proba(x))


# In[5]:


# AND
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[0],[0],[1]])
sgd = SGD(lr = 0.1)
model.compile(loss = 'binary_crossentropy',optimizer = 'sgd',metrics = ['accuracy'])
model.fit(x, y, epochs = 1000, batch_size = 1)
print(model.predict_proba(x))


# In[6]:


# iris
from builtins import range
import pandas as pd
datatrain = pd.read_csv('Datasets/iris/iris.csv')


# In[7]:


# 字串轉為數字
datatrain.loc[datatrain['species']=='Iris-setosa', 'species']=0
datatrain.loc[datatrain['species']=='Iris-versicolor', 'species']=1
datatrain.loc[datatrain['species']=='Iris-virginica', 'species']=2
datatrain = datatrain.apply(pd.to_numeric)


# In[8]:


# dataframe轉為array
datatrain_array = datatrain.values

# 分割x和y(feature and target)
xtrain = datatrain_array[:,:4]
ytrain = datatrain_array[:,4]

import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical

# 改變target形式
ytrain = to_categorical(ytrain) 


# In[9]:


# 建立模型
model = tf.keras.Sequential()
model.add(Dense(10, input_shape=(4,)))
model.add(Activation("relu"))
model.add(Dense(3))
model.add(Activation("softmax"))


# In[10]:


# 選擇優化和損失函數
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# 訓練
history = model.fit(xtrain, ytrain, epochs=300, batch_size=32)


# In[11]:


# 測試
datatest = pd.read_csv('Datasets/iris/iris_test.csv')

datatest.loc[datatest['species']=='Iris-setosa', 'species']=0
datatest.loc[datatest['species']=='Iris-versicolor', 'species']=1
datatest.loc[datatest['species']=='Iris-virginica', 'species']=2
datatest = datatest.apply(pd.to_numeric)

datatest_array = datatest.values

xtest = datatest_array[:,:4]
ytest = datatest_array[:,4]

ytest = to_categorical(ytest) 

score = model.evaluate(xtest, ytest, verbose = 0)

print('Test score:', score[0]*100)
print('Test accuracy:', score[1]*100,'%')


# In[12]:


# 預測
ypred = model.predict(xtest)

ytest_class = np.argmax(ytest, axis=1)
ypred_class = np.argmax(ypred, axis=1)

print(classification_report(ytest_class, ypred_class))
print(confusion_matrix(ytest_class, ypred_class))

