
# coding: utf-8

# In[1]:


#從 keras 的 datasets 匯入 mnist 資料集
from keras.datasets import mnist  
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# In[2]:


#神經網路架構
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))


# In[3]:


#編譯步驟
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# In[4]:


#準備圖片資料
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255


# In[5]:


#準備標籤
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[6]:


#檢驗神經網路模型
#加入驗證集validation_data
history = network.fit(train_images,
                      train_labels,
                      epochs=20,
                      batch_size=128,
                      validation_data=(test_images, test_labels))


# In[7]:


#評估測試資料的表現
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)


# In[8]:


#繪製訓練與驗證的損失函數
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']


epochs = range(1, len(loss_values)+ 1)

plt.plot(epochs, loss_values, 'g', label='Training loss', linewidth=3)
plt.plot(epochs, val_loss_values, 'b', label='Validation loss', linewidth=3)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()


# In[9]:


#繪製訓練和驗證的準確度
plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'g', label='Training acc', linewidth=3)
plt.plot(epochs, val_acc, 'b', label='Validation acc', linewidth=3)
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

