
# coding: utf-8

# In[1]:


from keras.datasets import mnist
from keras import models
from keras import layers
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


# In[7]:


def show_train_history(train_history, train, validation, epoch):  
    plt.plot(train_history.history[train], linewidth=3)  
    plt.plot(train_history.history[validation], linewidth=3)  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')  
    plt.legend(['Train', 'Validation'], loc='best')
    plt.grid(True)
    if train == 'acc':
        plt.savefig("epochs_acc_" + str(epoch) + ".jpg")
    if train == 'loss':
        plt.savefig("epochs_loss_" + str(epoch) + ".jpg")
    plt.show()  


# In[5]:


#epochs=5
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
train_history = network.fit(train_images, train_labels, validation_split=0.2, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)
#print(train_history.history)


# In[8]:


epoch = 5
show_train_history(train_history, 'acc', 'val_acc', 5)
show_train_history(train_history, 'loss', 'val_loss', 5)


# In[9]:


#評估模型準確率 
scores_5 = network.evaluate(train_images, train_labels)  
print()  
print("[Info] Accuracy of testing data = {:2.1f}%".format(scores_5[1]*100.0))  


# In[10]:


#epochs=20
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
train_history = network.fit(train_images, train_labels, validation_split=0.2, epochs=20, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)
#print(train_history.history)


# In[11]:


epoch = 20
show_train_history(train_history, 'acc', 'val_acc', 20)
show_train_history(train_history, 'loss', 'val_loss', 20)


# In[12]:


#評估模型準確率 
scores_20 = network.evaluate(train_images, train_labels)  
print()  
print("[Info] Accuracy of testing data = {:2.1f}%".format(scores_20[1]*100.0))  


# In[13]:


#epochs=50
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
train_history = network.fit(train_images, train_labels, validation_split=0.2, epochs=50, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)
#print(train_history.history)


# In[14]:


epoch = 50
show_train_history(train_history, 'acc', 'val_acc', 50)
show_train_history(train_history, 'loss', 'val_loss', 50)


# In[15]:


#評估模型準確率 
scores_50 = network.evaluate(train_images, train_labels)  
print()  
print("[Info] Accuracy of testing data = {:2.1f}%".format(scores_50[1]*100.0))  


# In[16]:


#epochs=100
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
train_history = network.fit(train_images, train_labels, validation_split=0.2, epochs=100, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)
#print(train_history.history)


# In[17]:


epoch = 100
show_train_history(train_history, 'acc', 'val_acc', 100)
show_train_history(train_history, 'loss', 'val_loss', 100)


# In[18]:


#評估模型準確率 
scores_100 = network.evaluate(train_images, train_labels)  
print()  
print("[Info] Accuracy of testing data = {:2.1f}%".format(scores_100[1]*100.0))  


# In[24]:


x = ['5','20','50','100']
accuracy = [round(scores_5[1]*100.0,2),round(scores_20[1]*100.0,2),round(scores_50[1]*100.0,2),round(scores_100[1]*100.0,2)]
plt.figure(figsize=(10,5))
plt.plot(x,accuracy,'r',lw=5)
plt.title('Accuracy of different epoch')
plt.ylabel('Accuracy',fontsize=16)
plt.xlabel('Epoch',fontsize=16)
plt.ylim(98,100)
plt.plot(x,accuracy,'blue',lw=5)
plt.scatter(x, accuracy, s = 75,color='b')
plt.xticks(range(4),['5','20','50','100'])
plt.grid(True)
for x,y in enumerate(accuracy):
    plt.text(x,y+0.11,'%s' %y, color='b', fontsize=16, ha='center')
plt.savefig('Accuracy of different epoch.jpg',dpi=300)

