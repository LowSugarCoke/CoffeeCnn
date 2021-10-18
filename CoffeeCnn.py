# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 16:01:36 2021

@author: Jack

Introduction:
    The CoffeeCnn is for coffee bean classification.
    
    1. Loading coffee bean image(train and test)
    2. Loading cls(train and test)
    3. Put data into cnn model
    4. Until the end of trainning
    4. Plot result

"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import CoffeeCnnLibs

def load_traindata(filename,number):
    begin = 0
    goodimages = np.zeros(shape=[80, 128, 128, 3], dtype=float)
    for i in range(number):
        test = np.array(Image.open('coffee bean/'+filename+str(i+1)+'.jpg'))
        test = test.reshape([-1, 3, 128, 128])
        test = test.transpose([0, 2, 3, 1])
        num_images = len(test)
        
        # End-index for the current batch.
        end = begin + num_images
        # Store the images into the array.
        goodimages[begin:end, :] = test
        begin = end
    return goodimages

def load_testdata(filename,number):
    begin = 0
    badimages = np.zeros(shape=[40, 128, 128, 3], dtype=float)
    for i in range(number):
        test = np.array(Image.open('coffee bean/'+filename+str(i+1)+'.jpg'))
        test = test.reshape([-1, 3, 128, 128])
        test = test.transpose([0, 2, 3, 1])
        num_images = len(test)
        
        # End-index for the current batch.
        end = begin + num_images
        # Store the images into the array.
        badimages[begin:end, :] = test
        begin = end
    return badimages


if __name__ == '__main__':
    train_images = np.zeros(shape=[80, 128, 128, 3], dtype=float)
    test_images = np.zeros(shape=[40, 128, 128, 3], dtype=float)

   
    train_images=load_traindata('train/train_',80)
    train_images=train_images/255
    
    test_images=load_testdata('test/test_',40)
    test_images=test_images/255  
    
    train_cls=np.loadtxt('coffee bean/train/traincls.txt',delimiter='	')
    test_cls=np.loadtxt('coffee bean/test/testcls.txt',delimiter='	')
    
    cnn_algorithm = CoffeeCnnLibs.CNNModel()
    cnn_algorithm.train(train_images,train_cls,test_images,test_cls,400)
    
    epoch_list = cnn_algorithm.get_epoch_list()
    loss_list = cnn_algorithm.get_loss_list()
    accuracy_list = cnn_algorithm.get_accuracy_list()
    duration = cnn_algorithm.get_duration()
    accuracy = cnn_algorithm.get_accuracy()
     
    #Plot 
    plt.figure(1)
    plt.plot(epoch_list,loss_list,'o-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['loss'])
    plt.figure(2)
    plt.plot(epoch_list,accuracy_list,'o-')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['accuracy'])     
    print("Train Finished takes:",duration)

    #Prediction Stage
    print("Accuracy:",accuracy)
    
