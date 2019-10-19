
# coding: utf-8

# In[1]:


from keras.datasets import mnist
from keras import models
from keras import layers
from keras.layers.core import Dense, Dropout, Activation
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("[Info] train data={:7,}".format(len(train_images)))  
print("[Info] test  data={:7,}".format(len(test_images)))  


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


def show_train_history(train_history, train, validation, node):  
    plt.plot(train_history.history[train], linewidth=3)  
    plt.plot(train_history.history[validation], linewidth=3)  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['Train', 'Validation'], loc='best')
    plt.grid(True)
    if train == 'acc':
        plt.savefig("hidden nodes_acc_" + str(node) + ".jpg")
    if train == 'loss':
        plt.savefig("hidden nodes_loss_" + str(node) + ".jpg")
    plt.show()  


# In[12]:


#32 nodes
network = models.Sequential()
network.add(layers.Dense(32, activation='relu', input_shape=(28 * 28,)))
network.add(Dropout(0.5))
network.add(layers.Dense(32, activation='relu'))
network.add(Dropout(0.5))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
train_history_32 = network.fit(train_images, train_labels, validation_split=0.2, epochs=20, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)
#print(train_history.history)

print("[Info] Model summary:")
network.summary()


# In[13]:


node = 32
show_train_history(train_history_32, 'acc', 'val_acc', node)
show_train_history(train_history_32, 'loss', 'val_loss', node)

scores_32 = network.evaluate(train_images, train_labels)
print()
print("[Info] Accuracy of testing data = {:2.1f}%".format(scores_32[1]*100.0)) 


# In[5]:


#64 nodes
network = models.Sequential()
network.add(layers.Dense(64, activation='relu', input_shape=(28 * 28,)))
network.add(Dropout(0.5))
network.add(layers.Dense(64, activation='relu'))
network.add(Dropout(0.5))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
train_history_64 = network.fit(train_images, train_labels, validation_split=0.2, epochs=20, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)
#print(train_history.history)

print("[Info] Model summary:")
network.summary()


# In[7]:


node = 64
show_train_history(train_history_64, 'acc', 'val_acc', node)
show_train_history(train_history_64, 'loss', 'val_loss', node)

scores_64 = network.evaluate(train_images, train_labels)
print()
print("[Info] Accuracy of testing data = {:2.1f}%".format(scores_64[1]*100.0)) 


# In[8]:


#128 nodes
network = models.Sequential()
network.add(layers.Dense(128, activation='relu', input_shape=(28 * 28,)))
network.add(Dropout(0.5))
network.add(layers.Dense(128, activation='relu'))
network.add(Dropout(0.5))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
train_history_128 = network.fit(train_images, train_labels, validation_split=0.2, epochs=20, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)
#print(train_history.history)

print("[Info] Model summary:")
network.summary()


# In[9]:


node = 128
show_train_history(train_history_128, 'acc', 'val_acc', node)
show_train_history(train_history_128, 'loss', 'val_loss', node)

scores_128 = network.evaluate(train_images, train_labels)
print()
print("[Info] Accuracy of testing data = {:2.1f}%".format(scores_128[1]*100.0)) 


# In[10]:


#256 nodes
network = models.Sequential()
network.add(layers.Dense(256, activation='relu', input_shape=(28 * 28,)))
network.add(Dropout(0.5))
network.add(layers.Dense(256, activation='relu'))
network.add(Dropout(0.5))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
train_history_256 = network.fit(train_images, train_labels, validation_split=0.2, epochs=20, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)
#print(train_history.history)

print("[Info] Model summary:")
network.summary()


# In[11]:


node = 256
show_train_history(train_history_256, 'acc', 'val_acc', node)
show_train_history(train_history_256, 'loss', 'val_loss', node)

scores_256 = network.evaluate(train_images, train_labels)
print()
print("[Info] Accuracy of testing data = {:2.1f}%".format(scores_256[1]*100.0)) 


# In[14]:


#512 nodes
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(Dropout(0.5))
network.add(layers.Dense(512, activation='relu'))
network.add(Dropout(0.5))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
train_history_512 = network.fit(train_images, train_labels, validation_split=0.2, epochs=20, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)
#print(train_history.history)

print("[Info] Model summary:")
network.summary()


# In[15]:


node = 512
show_train_history(train_history_512, 'acc', 'val_acc', node)
show_train_history(train_history_512, 'loss', 'val_loss', node)

scores_512 = network.evaluate(train_images, train_labels)
print()
print("[Info] Accuracy of testing data = {:2.1f}%".format(scores_512[1]*100.0)) 


# In[25]:


plt.plot(train_history_32.history['acc'], linewidth=3)
plt.plot(train_history_64.history['acc'], linewidth=3)
plt.plot(train_history_128.history['acc'], linewidth=3)
plt.plot(train_history_256.history['acc'], linewidth=3)
plt.plot(train_history_512.history['acc'], linewidth=3)
plt.title('Train History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])
plt.legend(['32 hidden nodes','64 hidden nodes','128 hidden nodes','256 hidden nodes','512 hidden nodes'], loc='best')
plt.grid(True)
plt.savefig('hidden_nodes_acc_all.jpg',dpi=300)


# In[26]:


plt.plot(train_history_32.history['loss'], linewidth=3)
plt.plot(train_history_64.history['loss'], linewidth=3)
plt.plot(train_history_128.history['loss'], linewidth=3)
plt.plot(train_history_256.history['loss'], linewidth=3)
plt.plot(train_history_512.history['loss'], linewidth=3)
plt.title('Train History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])
plt.legend(['32 hidden nodes','64 hidden nodes','128 hidden nodes','256 hidden nodes','512 hidden nodes'], loc='best')
plt.grid(True)
plt.savefig('hidden_nodes_loss_all.jpg',dpi=300)


# In[24]:


acc = [round(scores_32[1]*100,2),round(scores_64[1]*100,2),round(scores_128[1]*100,2),round(scores_256[1]*100,2),round(scores_512[1]*100,2)]
x = ['32','64','128','256','512']
plt.figure(figsize=(10,5))
plt.plot(x,acc,'b',lw=5)
plt.scatter(x, acc, s = 75,color='b')
plt.title('Accuracy of different hidden nodes', fontsize='18')
plt.xlabel('Hidden Nodes',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.grid(True)
for x,y in enumerate(acc):
    plt.text(x+0.1,y+0.25,'%s' %y, ha='right', color='b',fontsize=16)
plt.savefig('Accuracy of different hidden nodes.jpg',dpi=300)

