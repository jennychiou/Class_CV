
# coding: utf-8

# In[1]:


from keras.datasets import mnist
from keras import models
from keras import layers
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
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


def show_train_history(train_history, train, validation, size):  
    plt.plot(train_history.history[train], linewidth=3)  
    plt.plot(train_history.history[validation], linewidth=3)  
    plt.title('Train History')  
    plt.ylabel(train)  
    plt.xlabel('Epoch')
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])
    plt.legend(['Train', 'Validation'], loc='best')
    plt.grid(True)
    if train == 'acc':
        plt.savefig("batch_size_acc_" + str(size) + ".jpg")
    if train == 'loss':
        plt.savefig("batch_size_loss_" + str(size) + ".jpg")
    plt.show()  


# In[4]:


#batch_size=64
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
train_history_64 = network.fit(train_images, train_labels, validation_split=0.2, epochs=20, batch_size=64)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)


# In[5]:


size = 64
show_train_history(train_history_64, 'acc', 'val_acc', size)
show_train_history(train_history_64, 'loss', 'val_loss', size)

scores_64 = network.evaluate(train_images, train_labels)
print()
print("[Info] Accuracy of testing data = {:2.1f}%".format(scores_64[1]*100.0)) 


# In[6]:


#batch_size=128
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
train_history_128 = network.fit(train_images, train_labels, validation_split=0.2, epochs=20, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)


# In[7]:


size = 128
show_train_history(train_history_128, 'acc', 'val_acc', size)
show_train_history(train_history_128, 'loss', 'val_loss', size)

scores_128 = network.evaluate(train_images, train_labels)
print()
print("[Info] Accuracy of testing data = {:2.1f}%".format(scores_128[1]*100.0)) 


# In[8]:


#batch_size=256
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
train_history_256 = network.fit(train_images, train_labels, validation_split=0.2, epochs=20, batch_size=256)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)


# In[9]:


size = 256
show_train_history(train_history_256, 'acc', 'val_acc', size)
show_train_history(train_history_256, 'loss', 'val_loss', size)

scores_256 = network.evaluate(train_images, train_labels)
print()
print("[Info] Accuracy of testing data = {:2.1f}%".format(scores_256[1]*100.0)) 


# In[10]:


#batch_size=512
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
train_history_512 = network.fit(train_images, train_labels, validation_split=0.2, epochs=20, batch_size=512)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)


# In[11]:


size = 512
show_train_history(train_history_512, 'acc', 'val_acc', size)
show_train_history(train_history_512, 'loss', 'val_loss', size)

scores_512 = network.evaluate(train_images, train_labels)
print()
print("[Info] Accuracy of testing data = {:2.1f}%".format(scores_512[1]*100.0)) 


# In[12]:


#batch_size=1024
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
train_history_1024 = network.fit(train_images, train_labels, validation_split=0.2, epochs=20, batch_size=1024)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)


# In[13]:


size = 1024
show_train_history(train_history_1024, 'acc', 'val_acc', size)
show_train_history(train_history_1024, 'loss', 'val_loss', size)


scores_1024 = network.evaluate(train_images, train_labels)
print()
print("[Info] Accuracy of testing data = {:2.1f}%".format(scores_1024[1]*100.0)) 


# In[14]:


plt.plot(train_history_64.history['acc'], linewidth=3)
plt.plot(train_history_128.history['acc'], linewidth=3)
plt.plot(train_history_256.history['acc'], linewidth=3)
plt.plot(train_history_512.history['acc'], linewidth=3)
plt.plot(train_history_1024.history['acc'], linewidth=3)
plt.title('Train History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])
plt.legend(['batch size 64','batch size 128','batch size 256','batch size 512','batch size 1024'], loc='best')
plt.grid(True)
plt.savefig('batch_size_acc_all.jpg',dpi=300)


# In[15]:


plt.plot(train_history_64.history['loss'], linewidth=3)
plt.plot(train_history_128.history['loss'], linewidth=3)
plt.plot(train_history_256.history['loss'], linewidth=3)
plt.plot(train_history_512.history['loss'], linewidth=3)
plt.plot(train_history_1024.history['loss'], linewidth=3)
plt.title('Train History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20'])
plt.legend(['batch size 64','batch size 128','batch size 256','batch size 512','batch size 1024'], loc='best')
plt.grid(True)
plt.savefig('batch_size_loss_all.jpg',dpi=300)


# In[53]:


acc = [round(scores_64[1]*100,2),round(scores_128[1]*100,2),round(scores_256[1]*100,2),round(scores_512[1]*100,2),round(scores_1024[1]*100,2)]
x = ['64','128','256','512','1024']
plt.figure(figsize=(10,5))
plt.plot(x,acc,'b',lw=5)
plt.scatter(x, acc, s = 75,color='b')
plt.title('Accuracy of different batch size', fontsize='18')
plt.xlabel('Batch Size',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.grid(True)
for x,y in enumerate(acc):
    plt.text(x+0.3,y+0.15,'%s' %y, ha='right', color='b',fontsize=16)
plt.savefig('Accuracy of different batch size.jpg',dpi=300)

