#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.layers import Dense,Activation,Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from matplotlib.image import imread
import matplotlib.pyplot as plt
import tensorflow as tf

import numpy as np


# In[2]:


train = r'C:\Users\hh\OneDrive\Documents\Downloads\tomato\train'
test = r'C:\Users\hh\OneDrive\Documents\Downloads\tomato\val'


# In[3]:


trainDIR = r'C:\Users\hh\OneDrive\Documents\Downloads\tomato\train'
testDIR = r'C:\Users\hh\OneDrive\Documents\Downloads\tomato\val'


# In[4]:


# data size

size = 224
batch_size=32
epoch = 10


# In[5]:


# Data Augmentation
datagen = ImageDataGenerator(rescale=1./255, shear_range = 0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)

x_train = datagen.flow_from_directory(train, target_size=(size,size) ,batch_size=batch_size, class_mode='categorical', subset='training')
x_test = datagen.flow_from_directory(train, target_size=(size, size), batch_size=batch_size, class_mode='categorical', subset='validation')


x_test.class_indices.keys()


# In[6]:


# call back optimizer:

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

checkpoint=ModelCheckpoint(r'testmodel.h5',
                          monitor='val_loss',
                          mode='min',
                          save_best_only=True,
                          verbose=1)
earlystop=EarlyStopping(monitor='val_loss',
                       min_delta=0,
                       patience=50,
                       verbose=1,
                       restore_best_weights=True)

callbacks=[checkpoint,earlystop]


# In[7]:


model=Sequential()
model.add(Conv2D(64,(3,3),activation='relu',padding='same',input_shape=(size,size,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(10,activation='softmax'))


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])


# In[8]:


model.summary()


# In[9]:


history = model.fit(x = x_train, validation_data = x_test, epochs = epoch, callbacks = callbacks)


# In[11]:


acc = history.history['accuracy'] 
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, epoch+1)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

