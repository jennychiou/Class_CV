
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import itertools
import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.datasets import mnist
from keras.utils import to_categorical, np_utils 
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D  
from sklearn.metrics import confusion_matrix


# In[2]:


# train/test趨勢圖
def show_train_history(history, train, validation, modeltype, num, epochs):  
    plt.plot(history.history[train], linewidth=3)  
    plt.plot(history.history[validation], linewidth=3)  
    plt.title('Train History')
    my_x_ticks = np.arange(0,epochs,1)
    plt.xticks(my_x_ticks)
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['Train', 'Validation'], loc='best')
    plt.grid(True)
    if train == 'acc':
        plt.savefig("image/MNIST_acc_model_" + modeltype + str(num) + ".jpg", dpi=300)
    if train == 'loss':
        plt.savefig("image/MNIST_loss_model_" + modeltype + str(num) + ".jpg", dpi=300)
    plt.show()  


# # MLP

# In[3]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 將每一幅影像都轉換為一個長向量，大小為28*28=784
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# 將影像的畫素歸到0~1
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# In[4]:


# MLP Model 參數設定
modeltype = 'MLP'
optimizer = 'rmsprop'
batch_size = 128
num_classes = 10
epochs = 20
verbose = 1


# In[5]:


# 將類別向量轉換為二進制矩陣
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[6]:


model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.summary()


# In[7]:


model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
train_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(x_test, y_test))


# In[8]:


num = 1
score = model.evaluate(x_test, y_test, verbose=0)
show_train_history(train_history, 'acc', 'val_acc', modeltype, num, epochs)
show_train_history(train_history, 'loss', 'val_loss', modeltype, num, epochs)

df = pd.DataFrame()
metrics = ['Test loss','Test accuracy']
score = [score[0], score[1]]
df["metrics"] = metrics
df["score"] = score
df.set_index('metrics', inplace=True)
df


# # CNN

# In[9]:


np.random.seed(10)  
(x_train, y_train), (x_test, y_test) = mnist.load_data()  

x_train40 = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')  
x_test40 = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')  
print(x_train40.shape[0], 'train samples')
print(x_test40.shape[0], 'test samples')

x_train40_norm = x_train40 / 255  
x_test40_norm = x_test40 /255  
  
y_trainOneHot = np_utils.to_categorical(y_train)  
y_testOneHot = np_utils.to_categorical(y_test) 


# # CNN Model 1

# In[10]:


# CNN Model 1 參數設定
modeltype = 'CNN'
optimizer = 'sgd'
batch_size = 64
epochs = 20
verbose = 1


# In[11]:


model = Sequential()
model.add(Conv2D(16, (5,5), activation="relu", padding="same", data_format="channels_last", input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2), data_format="channels_last"))
model.add(Conv2D(36, (5,5), activation="relu", padding="same", data_format="channels_last"))
model.add(MaxPooling2D(pool_size=(2,2), data_format="channels_last"))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))
model.summary() 


# In[12]:


model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])  
train_history = model.fit(x=x_train40_norm, y=y_trainOneHot, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=verbose) 


# In[13]:


num = 1
score = model.evaluate(x_test40_norm, y_testOneHot, verbose=0)
show_train_history(train_history, 'acc', 'val_acc', modeltype, num, epochs)
show_train_history(train_history, 'loss', 'val_loss', modeltype, num, epochs)

df = pd.DataFrame()
metrics = ['Test loss','Test accuracy']
score = [score[0], score[1]]
df["metrics"] = metrics
df["score"] = score
df.set_index('metrics', inplace=True)
df


# # CNN Model 2

# In[14]:


# CNN Model 2 參數設定
modeltype = 'CNN'
optimizer = 'rmsprop'
batch_size = 128
epochs = 20
num_classes = 10
verbose = 1


# In[15]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu')) 
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()


# In[16]:


model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])  
train_history = model.fit(x=x_train40_norm, y=y_trainOneHot, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=(x_test40_norm, y_testOneHot)) 


# In[17]:


num = 2
score = model.evaluate(x_test40_norm, y_testOneHot, verbose=0)
show_train_history(train_history, 'acc', 'val_acc', modeltype, num, epochs)
show_train_history(train_history, 'loss', 'val_loss', modeltype, num, epochs)

df = pd.DataFrame()
metrics = ['Test loss','Test accuracy']
score = [score[0], score[1]]
df["metrics"] = metrics
df["score"] = score
df.set_index('metrics', inplace=True)
df


# # CNN Model 3

# In[18]:


# CNN Model 3 參數設定
modeltype = 'CNN'
optimizer = 'rmsprop'
batch_size = 128
epochs = 20
verbose = 1


# In[19]:


model = models.Sequential() 
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) # 加入 Covn2d 層 Conv2D(過濾器數量,過濾器長寬)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())  # 建立平坦層 
model.add(Dense(128, activation='relu'))  # 建立 Hidden layer 
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))  # 建立輸出層 
model.summary() 


# In[20]:


model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])  
train_history = model.fit(x=x_train40_norm, y=y_trainOneHot, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=verbose) 


# In[21]:


num = 3
score = model.evaluate(x_train40_norm, y_trainOneHot, verbose=0)
show_train_history(train_history, 'acc', 'val_acc', modeltype, num, epochs)
show_train_history(train_history, 'loss', 'val_loss', modeltype, num, epochs)

df = pd.DataFrame()
metrics = ['Test loss','Test accuracy']
score = [score[0], score[1]]
df["metrics"] = metrics
df["score"] = score
df.set_index('metrics', inplace=True)
df

