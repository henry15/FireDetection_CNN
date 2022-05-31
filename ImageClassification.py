#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Wed May 25 11:41:57 2022

@author: henry
"""

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Convolutional Neural Network

# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,MaxPooling2D,BatchNormalization,                        Permute, TimeDistributed, Bidirectional,GRU, SimpleRNN, LSTM, GlobalAveragePooling2D, SeparableConv2D
import os
import os.path
from pathlib import Path
#SCALER & TRANSFORMATION
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import regularizers
from sklearn.preprocessing import LabelEncoder

a = tf.__version__


Fire_Dataset_Path = Path("F:/ISM-Sem2/AI and ML/FireDetection-ImageClassification/Fire-DetectionImages")
image_Path = list(Fire_Dataset_Path.glob(r"*/*.jpg"))
image_Labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1],image_Path))
image_Path_Series = pd.Series(image_Path,name="JPG").astype(str)
image_Labels_Series = pd.Series(image_Labels,name="CATEGORY")
image_Labels_Series.replace({"0":"NO_FIRE","1":"FIRE"},inplace=True)

#TRANSFORMATION TO DATAFRAME STRUCTURE
Main_Train_Data = pd.concat([image_Path_Series,image_Labels_Series],axis=1)

#SHUFFLING
Main_Train_Data = Main_Train_Data.sample(frac=1).reset_index(drop=True)

# Part 1 - Data Preprocessing
# Preprocessing the Training set
# Below is Data augmentation technique
#Some of the most common data augmentation techniques used for images are:
#Position augmentation. Scaling. Cropping. Flipping. Padding. Rotation. Translation. Affine transformation.
#Color augmentation. Brightness. Contrast. Saturation. Hue.
train_datagenerator = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.3,
                                   zoom_range = 0.2,
                                   brightness_range=[0.2,0.9],
                                    rotation_range=30,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    fill_mode="nearest",
                                    validation_split=0.1)
test_datagenerator = ImageDataGenerator(rescale=1./255)

#SPLITTING TRAIN AND TEST
Train_Data,Test_Data = train_test_split(Main_Train_Data,train_size=0.9, random_state=42,shuffle=True)

encode = LabelEncoder()
For_Prediction_Class = encode.fit_transform(Test_Data["CATEGORY"])

#APPLYING GENERATOR AND TRANSFORMATION TO TENSOR
Train_IMG_Set = train_datagenerator.flow_from_dataframe(dataframe=Train_Data,
                                                   x_col="JPG",
                                                   y_col="CATEGORY",
                                                   color_mode="rgb",
                                                   class_mode="binary",
                                                   batch_size=32,
                                                   subset="training")

Validation_IMG_Set = train_datagenerator.flow_from_dataframe(dataframe=Train_Data,
                                                   x_col="JPG",
                                                   y_col="CATEGORY",
                                                   color_mode="rgb",
                                                   class_mode="binary",
                                                   batch_size=32,
                                                   subset="validation")

Test_IMG_Set = test_datagenerator.flow_from_dataframe(dataframe=Test_Data,
                                                 x_col="JPG",
                                                 y_col="CATEGORY",
                                                 color_mode="rgb",
                                                 class_mode="binary",
                                                 batch_size=32)

for data_batch,label_batch in Validation_IMG_Set:
    print("DATA SHAPE: ",data_batch.shape)
    print("LABEL SHAPE: ",label_batch.shape)
    break

# Part 2 - Building the CNN
# Initialising the CNN
cnn = tf.keras.models.Sequential()
# Step 1 - Convolution
cnn.add(Conv2D(32,(3,3),activation="relu", input_shape=(256,256,3)))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D((2,2))) # Step 2 - Pooling

# Adding a second convolutional layer
cnn.add(Conv2D(64,(3,3), activation="relu",padding="same"))
cnn.add(Dropout(0.2))  # compressing the image
cnn.add(MaxPooling2D((2,2)))

# Adding a third convolutional layer
cnn.add(Conv2D(128,(3,3), activation="relu",padding="same"))
cnn.add(Dropout(0.5))
cnn.add(MaxPooling2D((2,2)))

# Step 3 - Flattening. converts multi-dimensional matrix to single dimensional matrix
cnn.add(Flatten())

# dense layer requires input in single-dimensional shape
# Step 4 - Full Connection. Nonlinear fn is used as activation fn
cnn.add(Dense(256,activation="relu"))
cnn.add(Dropout(0.5))

# Step 5 - Output Layer
## For Binary Classification
cnn.add(Dense(1,activation="sigmoid"))
Call_Back = tf.keras.callbacks.EarlyStopping(monitor="loss",patience=5,mode="min")
cnn.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss="binary_crossentropy",metrics=["accuracy"])
CNN_Model = cnn.fit(Train_IMG_Set,validation_data = Validation_IMG_Set, epochs=20)

print(cnn.summary())


# In[2]:


Test_Results = cnn.evaluate(Test_IMG_Set)
display("LOSS:  " + "%.4f" % Test_Results[0])
display("ACCURACY:  " + "%.2f" % Test_Results[1])


# In[3]:



plt.plot(CNN_Model.history["accuracy"], label='train accuracy')
plt.plot(CNN_Model.history["val_accuracy"], label='val accuracy')
plt.ylabel("ACCURACY")
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[4]:


plt.plot(CNN_Model.history['loss'], label='train loss')
plt.plot(CNN_Model.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')


# In[5]:


PredictedData = cnn.predict(Test_IMG_Set)
#PredictedData = PredictedData.argmax(axis=-1)
test_pred = (PredictedData > 0.5).astype(int)
#print(test_pred)

fig, axes = plt.subplots(nrows=8,
                         ncols=8,
                         figsize=(20, 20),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(cv2.imread(Test_Data["JPG"].iloc[i]))
    txt = 'Fire'
    if(test_pred[i] == 0): txt = 'No Fire'
    ax.set_title(f"Actual :{Test_Data.CATEGORY.iloc[i]}\n PREDICTION :{ txt }")
plt.tight_layout()
plt.show()


# In[18]:


display(len( Test_Data[Test_Data["CATEGORY"] == "NO_FIRE"]))


# In[14]:


#ACCURACY CONTROL
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve
print(confusion_matrix(For_Prediction_Class,test_pred))


# In[20]:


(1+52)/(1+52+6+5)


# In[12]:


import numpy as np

#PredictClass = cnn.predict_classes(Test_IMG_Set)
#classes_x = PredictedData.argmax(PredictedData,axis=-1)
#predictions = (cnn.predict(PredictedData) > 0.5).astype("int32")
classes_x=np.argmax(PredictedData,axis=1)


# In[22]:



import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.image as mpimg

test_image = image.load_img('F:/ISM-Sem2/AI and ML/FireDetection-ImageClassification/SelfImages/2.jpg', 
                            target_size = (256,256))
arr = np.array(test_image)
arr=arr/255.0
#arr = arr.reshape(1,256,256,3)
arr = np.expand_dims(arr, axis = 0)
pred = cnn.predict(arr)
#pred = cnn(arr, training=False)
#pred = pred.argmax(axis=-1)
test_pred = (pred > 0.5).astype(int)

print(pred[0][0])
print(test_pred)

img = mpimg.imread('F:/ISM-Sem2/AI and ML/FireDetection-ImageClassification/SelfImages/2.jpg')
plt.imshow(img)
#img = Image.open('F:/ISM-Sem2/AI and ML/FireDetection-ImageClassification/SelfImages/1.jpg')

plt.title(f"Actual :NO FIRE \n PREDICTION : FIRE")
plt.tight_layout()
plt.show()


# In[ ]:




