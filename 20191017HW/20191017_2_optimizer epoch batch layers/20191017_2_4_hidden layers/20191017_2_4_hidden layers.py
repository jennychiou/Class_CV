
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


def show_train_history(train_history, train, validation, layer):  
    plt.plot(train_history.history[train], linewidth=3)  
    plt.plot(train_history.history[validation], linewidth=3)  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['Train', 'Validation'], loc='best')
    plt.grid(True)
    if train == 'acc':
        plt.savefig("hidden layers_acc_" + str(layer) + ".jpg")
    if train == 'loss':
        plt.savefig("hidden layers_loss_" + str(layer) + ".jpg")
    plt.show()  


# In[5]:


#1 Hidden Layers
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
train_history_1 = network.fit(train_images, train_labels, validation_split=0.2, epochs=20, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)
#print(train_history.history)

print("[Info] Model summary:")
network.summary()


# In[6]:


layer = 1
show_train_history(train_history_1, 'acc', 'val_acc', layer)
show_train_history(train_history_1, 'loss', 'val_loss', layer)

scores_1 = network.evaluate(train_images, train_labels)
print()
print("[Info] Accuracy of testing data = {:2.1f}%".format(scores_1[1]*100.0)) 


# In[7]:


#2 Hidden Layers
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dropout(0.5))
network.add(layers.Dense(128, activation='relu'))
network.add(layers.Dropout(0.5))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
train_history_2 = network.fit(train_images, train_labels, validation_split=0.2, epochs=20, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)
#print(train_history.history)

print("[Info] Model summary:")  
network.summary()


# In[8]:


layer = 2
show_train_history(train_history_2, 'acc', 'val_acc', layer)
show_train_history(train_history_2, 'loss', 'val_loss', layer)

scores_2 = network.evaluate(train_images, train_labels)
print()
print("[Info] Accuracy of testing data = {:2.1f}%".format(scores_2[1]*100.0)) 


# In[9]:


#3 Hidden Layers
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(Dropout(0.5))
network.add(layers.Dense(256, activation='relu'))
network.add(Dropout(0.5))
network.add(layers.Dense(128, activation='relu'))
network.add(Dropout(0.5))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
train_history_3 = network.fit(train_images, train_labels, validation_split=0.2, epochs=20, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)
#print(train_history.history)

print("[Info] Model summary:")  
network.summary()


# In[10]:


layer = 3
show_train_history(train_history_3, 'acc', 'val_acc', layer)
show_train_history(train_history_3, 'loss', 'val_loss', layer)

scores_3 = network.evaluate(train_images, train_labels)
print()
print("[Info] Accuracy of testing data = {:2.1f}%".format(scores_3[1]*100.0)) 


# In[11]:


plt.plot(train_history_1.history['acc'], linewidth=3)
plt.plot(train_history_2.history['acc'], linewidth=3)
plt.plot(train_history_3.history['acc'], linewidth=3)
plt.title('Train History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])
plt.legend(['1 hidden layers','2 hidden layers','3 hidden layers'], loc='best')
plt.grid(True)
plt.savefig('hidden_layers_acc_all.jpg',dpi=300)


# In[12]:


plt.plot(train_history_1.history['loss'], linewidth=3)
plt.plot(train_history_2.history['loss'], linewidth=3)
plt.plot(train_history_3.history['loss'], linewidth=3)
plt.title('Train History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])
plt.legend(['1 hidden layers','2 hidden layers','3 hidden layers'], loc='best')
plt.grid(True)
plt.savefig('hidden_layers_loss_all.jpg',dpi=300)


# In[13]:


acc = [round(scores_1[1]*100,2),round(scores_2[1]*100,2),round(scores_3[1]*100,2)]
x = ['1','2','3']
plt.figure(figsize=(10,5))
plt.plot(x,acc,'b',lw=5)
plt.scatter(x, acc, s = 75,color='b')
plt.title('Accuracy of different hidden layers', fontsize='18')
plt.xlabel('Hidden Layers',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.grid(True)
for x,y in enumerate(acc):
    plt.text(x,y+0.05,'%s' %y, ha='right', color='b',fontsize=16)
plt.savefig('Accuracy of different hidden layers.jpg',dpi=300)

