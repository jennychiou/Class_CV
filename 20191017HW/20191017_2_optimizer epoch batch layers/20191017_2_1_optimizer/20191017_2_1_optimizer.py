
# coding: utf-8

# In[1]:


from keras.datasets import mnist
from keras import models
from keras import layers
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# In[2]:


train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#優化器rmsprop
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
rmsprop_hist = network.fit(train_images, train_labels, epochs=5, batch_size=128)
rmsprop_test_loss, rmsprop_test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', rmsprop_test_loss)
print('test_acc:', rmsprop_test_acc)
print(rmsprop_hist.history)

plt.figure(figsize=(10,3))
plt.suptitle('RMSprop', fontsize=18)

plt.subplot(1,2,1)
plt.plot(rmsprop_hist.history['acc'], 'b', linewidth=3)
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.xticks([0, 1, 2, 3, 4], ['1', '2', '3', '4', '5'])
plt.grid(True)
plt.tight_layout()

plt.subplot(1,2,2)
plt.plot(rmsprop_hist.history['loss'], 'g', linewidth=3)
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.xticks([0, 1, 2, 3, 4], ['1', '2', '3', '4', '5'])
plt.grid(True)
plt.tight_layout()

plt.savefig('Accuracy of RMSprop.jpg',dpi=300)


# In[5]:


#優化器sgd
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
sgd_hist = network.fit(train_images, train_labels, epochs=5, batch_size=128)
sgd_test_loss, sgd_test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', sgd_test_loss)
print('test_acc:', sgd_test_acc)
print(sgd_hist.history)

plt.figure(figsize=(10,3))
plt.suptitle('SGD', fontsize=14)

plt.subplot(1,2,1)
plt.plot(sgd_hist.history['acc'], 'b', linewidth=3)
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.xticks([0, 1, 2, 3, 4], ['1', '2', '3', '4', '5'])
plt.grid(True)
plt.tight_layout()

plt.subplot(1,2,2)
plt.plot(sgd_hist.history['loss'], 'g', linewidth=3)
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.xticks([0, 1, 2, 3, 4], ['1', '2', '3', '4', '5'])
plt.grid(True)
plt.tight_layout()

plt.savefig('Accuracy of SGD.jpg',dpi=300)


# In[6]:


#優化器adam
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
adam_hist = network.fit(train_images, train_labels, epochs=5, batch_size=128)
adam_test_loss, adam_test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', adam_test_loss)
print('test_acc:', adam_test_acc)
print(adam_hist.history)

plt.figure(figsize=(10,3))
plt.suptitle('Adam', fontsize=14)

plt.subplot(1,2,1)
plt.plot(adam_hist.history['acc'], 'b', linewidth=3)
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.xticks([0, 1, 2, 3, 4], ['1', '2', '3', '4', '5'])
plt.grid(True)
plt.tight_layout()

plt.subplot(1,2,2)
plt.plot(adam_hist.history['loss'], 'g', linewidth=3)
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.xticks([0, 1, 2, 3, 4], ['1', '2', '3', '4', '5'])
plt.grid(True)
plt.tight_layout()

plt.savefig('Accuracy of Adam.jpg',dpi=300)


# In[7]:


#優化器adagrad
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
adagrad_hist = network.fit(train_images, train_labels, epochs=5, batch_size=128)
adagrad_test_loss, adagrad_test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', adagrad_test_loss)
print('test_acc:', adagrad_test_acc)
print(adagrad_hist.history)

plt.figure(figsize=(10,3))
plt.suptitle('AdaGrad', fontsize=14)

plt.subplot(1,2,1)
plt.plot(adagrad_hist.history['acc'], 'b', linewidth=3)
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.xticks([0, 1, 2, 3, 4], ['1', '2', '3', '4', '5'])
plt.grid(True)
plt.tight_layout()

plt.subplot(1,2,2)
plt.plot(adagrad_hist.history['loss'], 'g', linewidth=3)
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.xticks([0, 1, 2, 3, 4], ['1', '2', '3', '4', '5'])
plt.grid(True)
plt.tight_layout()

plt.savefig('Accuracy of AdaGrad.jpg',dpi=300)


# In[8]:


#優化器adadelta
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
adadelta_hist = network.fit(train_images, train_labels, epochs=5, batch_size=128)
adadelta_test_loss, adadelta_test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', adadelta_test_loss)
print('test_acc:', adadelta_test_acc)
print(adadelta_hist.history)

plt.figure(figsize=(10,3))
plt.suptitle('AdaDelta', fontsize=14)

plt.subplot(1,2,1)
plt.plot(adadelta_hist.history['acc'], 'b', linewidth=3)
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.xticks([0, 1, 2, 3, 4], ['1', '2', '3', '4', '5'])
plt.grid(True)
plt.tight_layout()

plt.subplot(1,2,2)
plt.plot(adadelta_hist.history['loss'], 'g', linewidth=3)
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.xticks([0, 1, 2, 3, 4], ['1', '2', '3', '4', '5'])
plt.grid(True)
plt.tight_layout()

plt.savefig('Accuracy of AdaDelta.jpg',dpi=300)


# In[9]:


#優化器nadam
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
nadam_hist = network.fit(train_images, train_labels, epochs=5, batch_size=128)
nadam_test_loss, nadam_test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', nadam_test_loss)
print('test_acc:', nadam_test_acc)
print(nadam_hist.history)

plt.figure(figsize=(10,3))
plt.suptitle('Nadam', fontsize=14)

plt.subplot(1,2,1)
plt.plot(nadam_hist.history['acc'], 'b', linewidth=3)
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.xticks([0, 1, 2, 3, 4], ['1', '2', '3', '4', '5'])
plt.grid(True)
plt.tight_layout()

plt.subplot(1,2,2)
plt.plot(nadam_hist.history['loss'], 'g', linewidth=3)
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.xticks([0, 1, 2, 3, 4], ['1', '2', '3', '4', '5'])
plt.grid(True)
plt.tight_layout()

plt.savefig('Accuracy of Nadam.jpg',dpi=300)


# In[10]:


# summarize history for accuracy
plt.figure(figsize=(8,8))
plt.plot(rmsprop_hist.history['acc'], linewidth=2)
plt.plot(sgd_hist.history['acc'], linewidth=2)
plt.plot(adam_hist.history['acc'], linewidth=2)
plt.plot(adagrad_hist.history['acc'], linewidth=2)
plt.plot(adadelta_hist.history['acc'], linewidth=2)
plt.plot(nadam_hist.history['acc'], linewidth=2)
plt.title('Model Accuracy', fontsize=18)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks([0, 1, 2, 3, 4], ['1', '2', '3', '4', '5'])
plt.legend(['RMSprop', 'SGD', 'Adam', 'AdaGrad', 'AdaDelta', 'Nadam'], loc='best')
plt.grid(True)
plt.savefig('Accuracy of different optimizer.jpg',dpi=300)


# In[11]:


# summarize history for loss
plt.figure(figsize=(8,8))
plt.plot(rmsprop_hist.history['loss'], linewidth=2)
plt.plot(sgd_hist.history['loss'], linewidth=2)
plt.plot(adam_hist.history['loss'], linewidth=2)
plt.plot(adagrad_hist.history['loss'], linewidth=2)
plt.plot(adadelta_hist.history['loss'], linewidth=2)
plt.plot(nadam_hist.history['loss'], linewidth=2)
plt.title('Model Loss', fontsize=18)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks([0, 1, 2, 3, 4], ['1', '2', '3', '4', '5'])
plt.legend(['RMSprop', 'SGD', 'Adam', 'AdaGrad', 'AdaDelta', 'Nadam'], loc='best')
plt.grid(True)
plt.savefig('Loss of different optimizer.jpg',dpi=300)

