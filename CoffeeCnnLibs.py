# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 15:30:52 2021
@author: Jack

Introduction:
    The CNN model is for coffee bean classification.
    
    initialize:
    cnn_algorithm = CoffeeCnnLibs.CNNModel()
    
    train:
    history, test_loss, test_accuracy,duration = cnn_algorithm.train(train_images,train_labels,test_images,test_labels,400)
    
    get result:
    print("Duration Time",duration)
    print("Test Accuracy",test_accuracy)
    plot_history(history)

    
"""
import tensorflow as tf
from tensorflow.keras import models, layers
from time import time

from tensorflow.python.keras.layers.core import Activation

class CNNModel:
    def _init_(self):
        return 
    
    def train(self,train_images,train_labels,test_images,test_labels,epochs):
         startTime = time()
         
         model = models.Sequential()
         model.add(layers.Conv2D(15,(5,5),activation='relu',input_shape=(128,128,3),padding= 'same'))
         model.add(layers.MaxPooling2D((2, 2),strides=(2,2),padding= 'same'))
         model.add(layers.Conv2D(30, (5, 5), activation='relu',padding= 'same'))
         model.add(layers.Flatten())
         model.add(layers.Dense(10, activation='relu'))
         model.add(layers.Dropout(0.2))
         model.add(layers.Dense(2, activation='softmax'))
         
         model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
         model.summary()
         
         
         history = model.fit(train_images, train_labels, epochs=epochs,verbose=2)
         
         test_loss, test_acc = model.evaluate(test_images, test_labels)
         
         duration = time()-startTime
         
         return history,test_loss, test_acc,duration
         
         

         
        



