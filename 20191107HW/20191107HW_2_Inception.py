
# coding: utf-8

# # InceptionV3

# In[1]:


import pandas as pd
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, array_to_img
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD
from keras import backend as K


# In[2]:


# 影像維度
img_width, img_height = 299, 299

train_data_dir = 'D:/Anaconda3/Scripts/5 上課資料/電腦視覺與人機互動/20191031HW/cats_and_dogs_small/train'
validation_data_dir = 'D:/Anaconda3/Scripts/5 上課資料/電腦視覺與人機互動/20191031HW/cats_and_dogs_small/validation'
nb_train_samples = 2000
nb_validation_samples = 800
batch_size = 2


# In[3]:


if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


# In[4]:


base_model = InceptionV3(weights='imagenet', include_top=False)

# 添加全域空間平均池化層
x = base_model.output
x = GlobalAveragePooling2D()(x)
# 增加全連階層
x = Dense(1024, activation='relu', name='fc1')(x)
prediction = Dense(2, activation='softmax', name='predictions')(x)
model = Model(inputs=base_model.input, outputs=prediction)
model.summary()


# In[5]:


# 凍結所有層(除了Bottleneck Layers以外)來進行微調
for layer in model.layers:
    if layer.name in ['predictions']:
        continue
    layer.trainable = False


# In[6]:


df = pd.DataFrame(([layer.name, layer.trainable] for layer in model.layers), columns=['layer', 'trainable'])
df


# In[7]:


train_datagen = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(directory='D:/Anaconda3/Scripts/5 上課資料/電腦視覺與人機互動/20191031HW/cats_and_dogs_small/train',
                                                    target_size=[img_width, img_height],
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

validation_datagen = ImageDataGenerator()
validation_generator = validation_datagen.flow_from_directory(directory='D:/Anaconda3/Scripts/5 上課資料/電腦視覺與人機互動/20191031HW/cats_and_dogs_small/validation',
                                                              target_size=[img_width, img_height],
                                                              batch_size=batch_size,
                                                              class_mode='categorical')


# In[8]:


model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = model.fit_generator(
        train_generator,
        steps_per_epoch = train_generator.samples // batch_size,
        validation_data = validation_generator,
        validation_steps = validation_generator.samples // batch_size,
        epochs = 20)

model.save('model-inceptionv3-final.h5')


# In[9]:


for i, layer in enumerate(base_model.layers):
    print(i, layer.name)
for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True


# In[10]:


from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(
        train_generator,
        steps_per_epoch = train_generator.samples // batch_size,
        validation_data = validation_generator,
        validation_steps = validation_generator.samples // batch_size,
        epochs = 20)

model.save('model-inceptionv3-final2.h5')


# In[11]:


from IPython.display import display
import matplotlib.pyplot as plt

X_val_sample, _ = next(validation_generator)
y_pred = model.predict(X_val_sample)

nb_sample = 4
for x, y in zip(X_val_sample[:nb_sample], y_pred[:nb_sample]):
    s = pd.Series({'Cat': 1-np.max(y), 'Dog': np.max(y)})
    axes = s.plot(kind='bar')
    axes.set_xlabel('Class')
    axes.set_ylabel('Probability')
    axes.set_ylim([0, 1])
    plt.show()

    img = array_to_img(x)
    display(img)


# In[12]:


scores = model.evaluate_generator(generator=validation_generator, steps=validation_generator.samples // batch_size)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (scores[0]/100, scores[1]*100))


# In[13]:


modelname = 'Inceptionv3'

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

metrics = 'acc'
filename = 'image/'+ modelname + '_' + metrics
plt.plot(epochs, acc, 'g', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(filename, dpi=300)
print('Save',filename)
plt.figure()

metrics = 'loss'
filename = 'image/'+ modelname + '_' + metrics
plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(filename, dpi=300)
print('Save',filename)
plt.show()

