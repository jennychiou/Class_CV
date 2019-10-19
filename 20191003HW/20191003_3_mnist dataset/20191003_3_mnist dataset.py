
# coding: utf-8

# In[1]:


import keras


# In[2]:


# 匯入資料
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# In[3]:


# 訓練資料
print('資料維度 : ',train_images.shape)
print('標籤個數 : ',len(train_labels))


# In[4]:


# 測試資料
print('資料維度 : ',test_images.shape)
print('標籤個數 : ',len(test_labels))


# In[5]:


from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))


# In[6]:


train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[7]:


network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[8]:


# 網路fit
network.fit(train_images, train_labels, epochs=5, batch_size=128)


# In[9]:


test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)


# In[10]:


# 修改優化器(sgd)
network.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)


# In[11]:


# 修改優化器(Adagrad)
network.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)


# In[12]:


# 修改優化器(Adam)
network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)


# In[13]:


# 修改隱藏層節點、層數
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(256, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(256, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

