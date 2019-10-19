
# coding: utf-8

# In[1]:


import numpy as np
from keras.layers import Dense
from keras.models import Sequential


# In[2]:


model = Sequential()


# In[3]:


model.add(Dense(units=2,activation='relu',input_dim=2))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[4]:


print(model.summary())
print(model.get_weights())


# In[5]:


# XOR
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

model.fit(x,y,epochs=1000,batch_size=4)
print(model.get_weights())
print(model.predict(x,batch_size=4))


# In[6]:


# OR
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,1])

model.fit(x,y,epochs=1000,batch_size=4)
print(model.get_weights())
print(model.predict(x,batch_size=4))


# In[7]:


# AND
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

model.fit(x,y,epochs=1000,batch_size=4)
print(model.get_weights())
print(model.predict(x,batch_size=4))


# In[8]:


# iris
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


iris = datasets.load_iris()
x = pd.DataFrame(iris['data'], columns=iris['feature_names'])
print("target_names: "+str(iris['target_names']))
y = pd.DataFrame(iris['target'], columns=['target'])
iris_data = pd.concat([x,y], axis=1)
iris_data.head()


# In[10]:


iris['target_names']


# In[11]:


target_name = {0:'setosa',1:'versicolor',2:'virginica'}
iris_data['target_name'] = iris_data['target'].map(target_name)
iris_data = iris_data[(iris_data['target_name'] == 'setosa')|(iris_data['target_name'] == 'versicolor')]
iris_data = iris_data[['sepal length (cm)','petal length (cm)','target_name']]
iris_data.head()


# In[12]:


target_class = {'setosa':1,'versicolor':-1}
iris_data['target_class'] = iris_data['target_name'].map(target_class)
del iris_data['target_name']
iris_data.head()


# In[13]:


def sign(z):
    if z > 0:
        return 1
    else:
        return -1


# In[14]:


w = np.array([0.,0.,0.])
error = 1
iterator = 0
while error != 0:
    error = 0
    for i in range(len(iris_data)):
        x,y = np.concatenate((np.array([1.]), np.array(iris_data.iloc[i])[:2])), np.array(iris_data.iloc[i])[2]
        if sign(np.dot(w,x)) != y:
            print("iterator: "+str(iterator))
            iterator += 1
            error += 1
            sns.lmplot('sepal length (cm)','petal length (cm)',data=iris_data, fit_reg=False, hue ='target_class')
            
            # 前一個Decision boundary 的法向量
            if w[1] != 0:
                x_last_decision_boundary = np.linspace(0,w[1])
                y_last_decision_boundary = (w[2]/w[1])*x_last_decision_boundary
                plt.plot(x_last_decision_boundary, y_last_decision_boundary,'c--')
            w += y*x            
            print("x: " + str(x))            
            print("w: " + str(w))
            # x向量 
            x_vector = np.linspace(0,x[1])
            y_vector = (x[2]/x[1])*x_vector
            plt.plot(x_vector, y_vector,'b')
            # Decision boundary 的方向向量
            x_decision_boundary = np.linspace(-0.5,7)
            y_decision_boundary = (-w[1]/w[2])*x_decision_boundary - (w[0]/w[2])
            plt.plot(x_decision_boundary, y_decision_boundary,'r')
            # Decision boundary 的法向量
            x_decision_boundary_normal_vector = np.linspace(0,w[1])
            y_decision_boundary_normal_vector = (w[2]/w[1])*x_decision_boundary_normal_vector
            plt.plot(x_decision_boundary_normal_vector, y_decision_boundary_normal_vector,'g')
            plt.xlim(-0.5,7.5)
            plt.ylim(5,-3)
            plt.show()

