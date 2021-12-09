# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 16:01:36 2021

@author: Jack

Introduction:
    The CoffeeCnn is for coffee bean classification.
    
    1. Loading coffee bean image(train and test)
    2. Loading label(train and test)
    3. Put data into cnn model
    4. Until the end of trainning
    4. Plot result

"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import CoffeeCnnLibs
import pandas as pd


def load_traindata(filename,number):
    begin = 0
    train_images = np.zeros(shape=[80, 128, 128, 3], dtype=float)
    for i in range(number):
        temp = np.array(Image.open('data/'+filename+str(i+1)+'.jpg'))
        temp = temp.reshape([-1, 3, 128, 128])
        temp = temp.transpose([0, 2, 3, 1])
        num_images = len(temp)
        
        # End-index for the current batch.
        end = begin + num_images
        # Store the images into the array.
        train_images[begin:end, :] = temp
        begin = end
    return train_images

def load_testdata(filename,number):
    begin = 0
    test_images = np.zeros(shape=[40, 128, 128, 3], dtype=float)
    for i in range(number):
        temp = np.array(Image.open('data/'+filename+str(i+1)+'.jpg'))
        temp = temp.reshape([-1, 3, 128, 128])
        temp = temp.transpose([0, 2, 3, 1])
        num_images = len(temp)
        
        # End-index for the current batch.
        end = begin + num_images
        # Store the images into the array.
        test_images[begin:end, :] = temp
        begin = end
    return test_images

def plot_history(history):
    
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.plot(hist['epoch'], hist['loss'],
           label='Loss')
  plt.plot(hist['epoch'], hist['accuracy'],
           label = 'Accuracy')
  plt.ylim([0,1])
  plt.legend()

  plt.show()

if __name__ == '__main__':
    train_images = np.zeros(shape=[80, 128, 128, 3], dtype=float)
    test_images = np.zeros(shape=[40, 128, 128, 3], dtype=float)


    train_images=load_traindata('train/train_',80)    
    train_images=train_images/255.0
    
    test_images=load_testdata('test/test_',40)    
    test_images=test_images/255.0  
    
    train_labels=np.loadtxt('data/train/traincls.txt')
    test_labels=np.loadtxt('data/test/testcls.txt')
    
    
    cnn_algorithm = CoffeeCnnLibs.CNNModel()
    history, test_loss, test_accuracy,duration = cnn_algorithm.train(train_images,train_labels,test_images,test_labels,400)
    
    print("Duration Time",duration)
    print("Test Accuracy",test_accuracy)
    
    plot_history(history)
    
    
   
    
   