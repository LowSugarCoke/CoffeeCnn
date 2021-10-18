# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 15:30:52 2021
@author: Jack

Introduction:
    The CNN model is for coffee bean classification.
    
    initialize:
    cnn_algorithm = CoffeeCnnLibs.CNNModel()
    
    train:
    cnn_algorithm.train(goodimages,goodcls,badimages,badcls,400)
    
    get result:
    epoch_list = cnn_algorithm.getepochlist()
    loss_list = cnn_algorithm.getlosslist()
    accuracy_list = cnn_algorithm.getaccuracylist()
    duration = cnn_algorithm.getduration()
    accuracy = cnn_algorithm.getaccuracy()
    
"""
import tensorflow.compat.v1 as tf
from time import time

class CNNModel:
    def _init_(self):
        self.epoch_list = []
        self.loss_list = []
        self.accuracy_list = []
        self.duration = 0
        self.accuracy = 0
        return 
    
    def get_epoch_list(self):
        return self.epoch_list
    
    def get_loss_list(self):
        return self.loss_list
    
    def get_accuracy_list(self):
        return self.accuracy_list
    
    def get_duration(self):
        return self.duration
    
    def get_accuracy(self):
        return self.accuracy
        
        #Step1
    def weight(self,shape):
        return tf.Variable(tf.truncated_normal(shape,stddev=0.1),name='W')
        #Step2
    def bias(self,shape):
        return(tf.Variable(tf.constant(0.1,shape=shape),name='b'))
        #Step3
    def conv2d(self,x,W):
        return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
    #Step4
    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')      
    
    def train(self,goodimages,goodcls,badimages,badcls,times):
        tf.disable_v2_behavior()
        tf.logging.set_verbosity(tf.logging.ERROR)
         #Modeling Stage
         #Step1
        with tf.name_scope('Input_Layer'):
            x = tf.placeholder("float",shape=[None,128,128,3])
            x_image =tf.reshape(x,[-1,128,128,3])
        #Step2
        with tf.name_scope('C1_Conv'):
            W1 = self.weight([5,5,3,15])
            b1 = self.bias([15])
            Conv1 = self.conv2d(x_image,W1)+b1
            C1_Conv= tf.nn.relu(Conv1)
        #Step3
        with tf.name_scope('Cl_Pool'):
            C1_Pool = self.max_pool_2x2(C1_Conv)
        #Step4
        with tf.name_scope('C2_Conv'):
            W2 = self.weight([5,5,15,30])
            b2 = self.bias([30])
            Conv2 = self.conv2d(C1_Pool,W2)+b2
            C2_Conv=tf.nn.relu(Conv2)
        #Step5
        with tf.name_scope('D_Flat'):
            D_Flat = tf.reshape(C2_Conv,[-1,122880])
        #Step6
        with tf.name_scope('D_Hidden_Layer'):
            W3 = self.weight([122880,10])
            b3 = self.bias([10])
            D_Hidden = tf.nn.relu(tf.matmul(D_Flat,W3)+b3)
            D_Hidden_Dropout = tf.nn.dropout(D_Hidden,keep_prob=0.8)
        #Step7
        with tf.name_scope('Output_layer'):
            W4 = self.weight([10,2])
            b4 = self.bias([2])
            y_predict = tf.nn.softmax(tf.matmul(D_Hidden_Dropout,W4)+b4)

        #Definition of Training Algorithm Stage
        with tf.name_scope('optimizer'):
            y_label = tf.placeholder("float",shape=[None,2],name="y_label")
            loss_function=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict,labels=y_label))
            optimizer=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_function)

        with tf.name_scope("evaluate_model"):
            correct_prediction = tf.equal(tf.argmax(y_predict,1),tf.argmax(y_label,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
               
        #Training Stage
        trainEpoch = times
        totalBatchs = 1
        self.epoch_list=[]
        self.accuracy_list=[]
        self.loss_list=[]

        startTime = time()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for epoch in range(trainEpoch):
            for i in range(totalBatchs):
                batch_x=goodimages
                batch_y=goodcls
                sess.run(optimizer,feed_dict={x:batch_x,y_label:batch_y})
        
            loss,acc=sess.run([loss_function,accuracy],feed_dict={x:badimages,y_label:badcls})
            self.epoch_list.append(epoch)
            self.loss_list.append(loss)
            self.accuracy_list.append(acc)
            print("Train Epoch:",'%02d'%(epoch+1),"Loss=","{:.9f}".format(loss),"Accuracy=",acc)
            
        self.duration = time()-startTime
        self.accuracy = sess.run(accuracy,feed_dict={x:badimages,y_label:badcls})
        return 



