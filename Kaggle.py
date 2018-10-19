#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import imutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.preprocessing import image
from PIL import Image, ImageDraw


# In[6]:


path = '/media/vlados/FreeSpace/Kaggle/train_simple' #csv files path

def draw_it(raw_strokes):
        image_ = Image.new("P", (255,255), color=255)
        image_draw = ImageDraw.Draw(image_)

        for stroke in eval(raw_strokes):
            for i in range(len(stroke[0])-1):

                image_draw.line([stroke[0][i], 
                                 stroke[1][i],
                                 stroke[0][i+1], 
                                 stroke[1][i+1]],
                                fill=0, width=6)
        image_ = np.array(image_) 
        image_ = imutils.resize(image_, width=32)
        image_ = image_[:,:,np.newaxis]
        
        return (image_)

def dataset_maker(path, train_img, test_img):
    
    iterator = 0
    classes_name = os.listdir(path)
    
    image_in_test_dataset = test_img
    image_in_class = train_img + test_img
    
    num_test_samples = image_in_test_dataset * len(classes_name)
    num_train_samples = image_in_class * len(classes_name)
    
    X_train = np.zeros((num_train_samples, 32, 32, 1), dtype='uint16')
    Y_train = np.zeros((num_train_samples), dtype='uint16')
    
    x_test = np.zeros((num_test_samples, 32, 32, 1), dtype='uint16')
    y_test = np.zeros((num_test_samples, 1), dtype='uint16')
    
    all_data_in_class = pd.read_csv(path + '/' + classes_name[0])
    all_data_in_class = all_data_in_class.values
        
    for i in range(0, len(classes_name) - 1):
        all_data_in_class = pd.read_csv(path + '/' + classes_name[i])
        all_data_in_class = all_data_in_class.values
        
        iterator = i * image_in_class

        for img in range(0, image_in_class - 1):
            
            if image_in_class < train_img:
                X_train[img + iterator,:,:,:] = draw_it(all_data_in_class[img,1])
                Y_train[img + iterator] = i
            
            else:
                x_test[i,:,:,:] = draw_it(all_data_in_class[img,1])
                y_test[i] = i
                i = i + 1
    
    randomize_train = np.arange(num_train_samples)
    randomize_test = np.arange(num_test_samples)
    
    np.random.shuffle(randomize_train)
    np.random.shuffle(randomize_test)
    
    X_train = X_train[randomize_train]
    Y_train = Y_train[randomize_train]
    
    x_test = x_test[randomize_test]
    y_test = y_test[randomize_test]

    return (X_train, Y_train), (x_test, y_test)


# In[7]:


(X_train, Y_train), (x_test, y_test) = dataset_maker(path, 5000, 1000)


# In[9]:


np.save('/media/vlados/FreeSpace/Kaggle/dataset/X_train.npy', X_train)
np.save('/media/vlados/FreeSpace/Kaggle/dataset/Y_train.npy', Y_train)
np.save('/media/vlados/FreeSpace/Kaggle/dataset/x_test.npy', x_test)
np.save('/media/vlados/FreeSpace/Kaggle/dataset/y_test.npy', y_test)


# In[15]:


y_test = None


# In[16]:


y_test = np.load('/media/vlados/FreeSpace/Kaggle/dataset/y_test.npy')

