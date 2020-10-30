#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras

from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

from sklearn.model_selection import KFold

from sklearn import metrics
from sklearn.metrics import accuracy_score,matthews_corrcoef,classification_report,confusion_matrix,precision_score,recall_score
from sklearn.metrics import f1_score,roc_auc_score, auc

from keras import regularizers

import os
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from keras.utils import plot_model




import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from tensorflow.python.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示error和warining信息 3 只显示error信息
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 这一行注释掉就是使用cpu，不注释就是使用gpu


# In[ ]:





# In[2]:


def RESNET_pca15(Dropout1=0,Epochs= 20,Batch_size=64,PCA_num = 15):
    # 优化器选择 Adam 优化器。
    # 损失函数使用 sparse_categorical_crossentropy，
    # 还有一个损失函数是 categorical_crossentropy，两者的区别在于输入的真实标签的形式，
    # sparse_categorical 输入的是整形的标签，例如 [1, 2, 3, 4]，categorical 输入的是 one-hot 编码的标签。
    

    train_Feature = np.load("E:/yanyi/CDR3/process/net_resnet/data/train_TCRA_PCA{}_feature_array.npy".format(PCA_num))    
    train_Label = np.load("E:/yanyi/CDR3/process/net_resnet/data/train_TCRA_PCA{}_label_array.npy".format(PCA_num))
    
    test_Feature = np.load("E:/yanyi/CDR3/process/net_resnet/data/test_TCRA_PCA{}_feature_array.npy".format(PCA_num)) 
    test_Label = np.load("E:/yanyi/CDR3/process/net_resnet/data/test_TCRA_PCA{}_label_array.npy".format(PCA_num))
      
           
    X_train = train_Feature
    Y_train = train_Label#[:,1]  
           
    X_test = test_Feature
    Y_test = test_Label#[:,1]  
    
    
    X_train,Y_train = shuffle(X_train,Y_train)
    X_test,Y_test = shuffle(X_test,Y_test)

    
    info = "using the one_hoot th model\n"
    modelfile = './RESNET_A_fold_pca15_yes2.h5'

    X_train= X_train.reshape([len(X_train),20,PCA_num+1,2])
    X_test = X_test.reshape([len(X_test),20,PCA_num+1,2])
    X_test=tf.cast(X_test, tf.float32)
    
    

    
    Y_pred = resnet_attention_train_predict(20, PCA_num ,modelfile,info,2,X_train, Y_train,X_test, Y_test)


    Y_pred1 = np.argmax(Y_pred, axis=-1)
    Y_test1 = np.argmax(Y_test, axis=-1)



    
    
    #print(Y_pred)
    confusion_matrix1 =confusion_matrix(Y_test1,Y_pred1)

    accuracy = accuracy_score(Y_test1,Y_pred1) #准确率
    precision = precision_score(Y_test1,Y_pred1) #精确率
    recall = recall_score(Y_test1,Y_pred1) #召回率
    f1= f1_score(Y_test1,Y_pred1) #F1
    MCC = matthews_corrcoef(Y_test1,Y_pred1) #MCC

   
    
    print('混淆矩阵\n',confusion_matrix1,
          '\n准确率ACC:',accuracy,
          '\n精确率precision:',precision,
          '\n召回率recall:',recall,
          '\nF1:',f1,
          '\nMCC:',MCC 
         )
               


    #print(y_probs)

    fpr, tpr, thresholds = metrics.roc_curve(Y_test[:,1], Y_pred[:,1])
    roc_auc = auc(fpr, tpr)  #auc为Roc曲线下的面积
    #开始画ROC曲线
    plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.xlabel('False Positive Rate') #横坐标是fpr
    plt.ylabel('True Positive Rate')  #纵坐标是tpr
    plt.title('Receiver operating characteristic example')
    plt.show()


                              

    


# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 12:15:27 2020

@author: Administrator
"""

import pandas
import numpy as np
import os

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import f1_score,roc_auc_score,recall_score,precision_score

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint




def lr_schedule(epoch):
    lr = 1e-3
    return lr*0.9*epoch


def resnet_layer(inputs, num_filters, kernel_size=3, strides=1,
                 activation='relu', batch_normalization=True, conv_first=True):
    ''' 2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    '''
    conv = layers.Conv2D(num_filters, kernel_size=kernel_size, strides=strides,
                         padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(1e-4))
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
        x = conv(x)
        
    return x

def resnet_v1(input_shape, depth, num_classes=2):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth-2)%6 != 0:
        raise ValueError('depth should be 6n+2')
    # Start model definition.
    num_filters = 32
    num_res_blocks = int((depth-2)/6)
    
    inputs = tf.keras.Input(shape=input_shape)
    x = resnet_layer(inputs, num_filters)
    # Instantiate teh stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0: # first layer but not first stack
                strides = 2 # downsample
            y = resnet_layer(x, num_filters, strides=strides)  
            y = resnet_layer(y, num_filters, activation=None)
            
            if stack > 0 and res_block == 0: # first layer but not first stack
                # linear projection residual shortcut connection to match
                # change dims
                x = resnet_layer(x, num_filters, kernel_size=1, strides=strides,
                                 activation=None, batch_normalization=False)
            x = layers.add([x, y])
            x = layers.Activation('relu')(x)           
        num_filters *= 2
        
    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    ax = layers.GlobalAveragePooling2D()(x)
    #x = layers.AveragePooling2D()(x)
    
    ax = layers.Dense(num_filters//8, activation='relu')(ax)
    ax = layers.Dense(num_filters//2, activation='softmax')(ax)
    ax = layers.Reshape((1,1,num_filters//2))(ax)
    ax = layers.Multiply()([ax, x])
    y = layers.Flatten()(ax)
    outputs = layers.Dense(num_classes, activation='softmax',
                           kernel_initializer='he_normal')(y)
    # Instantiate model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model   


def resnet_attention_train_predict(row, col, modelfile, info, m, x_train, y_train,x_test, y_test):
    y_train, y_train,x_test, y_test = x_train, y_train,x_test, y_test
    
    
    
    model = resnet_v1(input_shape=(row, col+1, m), depth=20, num_classes=2)
    
    
    
    
#     model.compile(optimizer=Adam(learning_rate=lr_schedule(0)),
#                  loss='categorical_crossentropy',
#                  metrics=['accuracy'])
    
#     model.compile(optimizer=Adam,
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])
    
    model.compile(optimizer="Adam",
              loss=keras.losses.binary_crossentropy,
              metrics=['accuracy'])
        
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5,
                                   min_lr=0.5e-6)

    checkpoint = ModelCheckpoint(filepath=modelfile, monitor='val_loss',
                                verbose=0, save_best_only=True)#,save_weights_only=True)
    
    cbs = [checkpoint, lr_reducer, lr_scheduler]
    model.fit(x_train, y_train,
                  batch_size=32,
                  epochs=50,
                  verbose=1, 
                  #validation_split=0.1,
                  validation_data=[x_test, y_test],
                  shuffle=False,
                  callbacks=cbs)#callbacks=cbs
    
    model.load_weights(modelfile)  
    
    y_pred = model.predict(x_test)  
    #y_pred = model.predict_classes(x_test)
#     y_predict = model.predict(x_test)      

#     y_probs = model.predict_proba(x_test) #模型的预测得分
#     #print(y_probs)

#     fpr, tpr, thresholds = metrics.roc_curve(y_test,y_probs)
#     roc_auc = auc(fpr, tpr)  #auc为Roc曲线下的面积
#     #开始画ROC曲线
#     plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
#     plt.legend(loc='lower right')
#     plt.plot([0,1],[0,1],'r--')
#     plt.xlim([-0.1,1.1])
#     plt.ylim([-0.1,1.1])
#     plt.xlabel('False Positive Rate') #横坐标是fpr
#     plt.ylabel('True Positive Rate')  #纵坐标是tpr
#     plt.title('Receiver operating characteristic example')
#     plt.show()
    model.save('./RESNET_A_fold_pca15_10.h5')
    
    del model
    
    return y_pred

    


# In[ ]:





# In[ ]:





# In[4]:


RESNET_pca15(0.3,5,128,15)


# In[5]:


#0
# 混淆矩阵
#  [[ 935  270]
#  [ 157 1048]] 
# 准确率ACC: 0.8228215767634854 
# 精确率precision: 0.795144157814871 
# 召回率recall: 0.8697095435684647 
# F1: 0.8307570352754657 
# MCC: 0.6485008843732686

# 混淆矩阵
#  [[ 720  485]
#  [  55 1150]] 
# 准确率ACC: 0.7759336099585062 
# 精确率precision: 0.7033639143730887 
# 召回率recall: 0.9543568464730291 
# F1: 0.8098591549295775 
# MCC: 0.5907611341052326


# 2
# 混淆矩阵
#  [[988 217]
#  [210 995]] 
# 准确率ACC: 0.8228215767634854 
# 精确率precision: 0.820957095709571 
# 召回率recall: 0.8257261410788381 
# F1: 0.8233347124534547 
# MCC: 0.645654047731702

# 3
# 混淆矩阵
#  [[ 925  280]
#  [ 176 1029]] 
# 准确率ACC: 0.8107883817427386 
# 精确率precision: 0.786096256684492 
# 召回率recall: 0.8539419087136929 
# F1: 0.818615751789976 
# MCC: 0.6239048115767343

# 4
# 混淆矩阵
#  [[ 934  271]
#  [ 170 1035]] 
# 准确率ACC: 0.8170124481327801 
# 精确率precision: 0.7924961715160797 
# 召回率recall: 0.8589211618257261 
# F1: 0.8243727598566308 
# MCC: 0.6362638271016609

# 5
# 混淆矩阵
#  [[ 927  278]
#  [ 158 1047]] 
# 准确率ACC: 0.8190871369294606 
# 精确率precision: 0.790188679245283 
# 召回率recall: 0.8688796680497926 
# F1: 0.8276679841897233 
# MCC: 0.6413624529216982

# 6
# 混淆矩阵
#  [[ 813  392]
#  [ 148 1057]] 
# 准确率ACC: 0.7759336099585062 
# 精确率precision: 0.7294685990338164 
# 召回率recall: 0.8771784232365145 
# F1: 0.7965335342878673 
# MCC: 0.5635413119881741

# 7
# 混淆矩阵
#  [[ 939  266]
#  [ 178 1027]] 
# 准确率ACC: 0.8157676348547718 
# 精确率precision: 0.794276875483372 
# 召回率recall: 0.8522821576763485 
# F1: 0.8222578062449959 
# MCC: 0.6332261009890243

# 8
# 混淆矩阵
#  [[1015  190]
#  [ 228  977]] 
# 准确率ACC: 0.8265560165975103 
# 精确率precision: 0.8371893744644388 
# 召回率recall: 0.8107883817427386 
# F1: 0.8237774030354132 
# MCC: 0.6534370268316719

9


# In[6]:




model1 = load_model('./RESNET_A_fold_pca15_yes2.h5')

Feature_test1 = np.load("E:/yanyi/CDR3/process/net_resnet/data/SARS-CoV-2_TCRA_PCA15_feature_array.npy")    
Label_array1 = np.load("E:/yanyi/CDR3/process/net_resnet/data/SARS-CoV-2_TCRA_PCA15_label_array.npy")

X_TEST = Feature_test1

Y_TEST = Label_array1
X_TEST = X_TEST.reshape([len(X_TEST),20,16,2])


Y_pred = model1.predict(X_TEST)



Y_pred1 = np.argmax(Y_pred, axis=-1)
Y_TEST1 = np.argmax(Y_TEST, axis=-1)
#print(Y_pred)




confusion_matrix1 =confusion_matrix(Y_TEST1,Y_pred1)

accuracy = accuracy_score(Y_TEST1,Y_pred1) #准确率
precision = precision_score(Y_TEST1,Y_pred1) #精确率
recall = recall_score(Y_TEST1,Y_pred1) #召回率
f1= f1_score(Y_TEST1,Y_pred1) #F1
MCC = matthews_corrcoef(Y_TEST1,Y_pred1) #MCC


print('混淆矩阵\n',confusion_matrix1,
      '\n准确率ACC:',accuracy,
      '\n精确率precision:',precision,
      '\n召回率recall:',recall,
      '\nF1:',f1,
      '\nMCC:',MCC 
     )


#print(y_probs)

fpr, tpr, thresholds = metrics.roc_curve(Y_TEST[:,1],Y_pred[:,1])
roc_auc = auc(fpr, tpr)  #auc为Roc曲线下的面积
#开始画ROC曲线
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.xlabel('False Positive Rate') #横坐标是fpr
plt.ylabel('True Positive Rate')  #纵坐标是tpr
plt.title('Receiver operating characteristic example')
plt.show()


# In[ ]:





# In[7]:


#0
# 混淆矩阵
#  [[227  65]
#  [ 61 231]] 
# 准确率ACC: 0.7842465753424658 
# 精确率precision: 0.7804054054054054 
# 召回率recall: 0.791095890410959 
# F1: 0.7857142857142858 
# MCC: 0.5685464977643606


#1
# 混淆矩阵
#  [[182 110]
#  [ 44 248]] 
# 准确率ACC: 0.7363013698630136 
# 精确率precision: 0.6927374301675978 
# 召回率recall: 0.8493150684931506 
# F1: 0.7630769230769231 
# MCC: 0.48515817476464324

# 2
# 混淆矩阵
#  [[244  48]
#  [ 73 219]] 
# 准确率ACC: 0.7928082191780822 
# 精确率precision: 0.8202247191011236 
# 召回率recall: 0.75 
# F1: 0.7835420393559929 
# MCC: 0.5877746460061437

# 3
# 混淆矩阵
#  [[238  54]
#  [ 39 253]] 
# 准确率ACC: 0.8407534246575342 
# 精确率precision: 0.8241042345276873 
# 召回率recall: 0.8664383561643836 
# F1: 0.8447412353923205 
# MCC: 0.6824078344349994

# 4
# 混淆矩阵
#  [[229  63]
#  [ 81 211]] 
# 准确率ACC: 0.7534246575342466 
# 精确率precision: 0.7700729927007299 
# 召回率recall: 0.7226027397260274 
# F1: 0.745583038869258 
# MCC: 0.507815072510736

# 5
# 混淆矩阵
#  [[230  62]
#  [130 162]] 
# 准确率ACC: 0.6712328767123288 
# 精确率precision: 0.7232142857142857 
# 召回率recall: 0.5547945205479452 
# F1: 0.6279069767441859 
# MCC: 0.35214760613688195

# 6
# 混淆矩阵
#  [[179 113]
#  [ 63 229]] 
# 准确率ACC: 0.6986301369863014 
# 精确率precision: 0.6695906432748538 
# 召回率recall: 0.7842465753424658 
# F1: 0.7223974763406941 
# MCC: 0.40321553225735096

# 7
# 混淆矩阵
#  [[222  70]
#  [ 80 212]] 
# 准确率ACC: 0.7431506849315068 
# 精确率precision: 0.75177304964539 
# 召回率recall: 0.726027397260274 
# F1: 0.7386759581881532 
# MCC: 0.4865867948660844

# 8
# 混淆矩阵
#  [[222  70]
#  [ 78 214]] 
# 准确率ACC: 0.7465753424657534 
# 精确率precision: 0.7535211267605634 
# 召回率recall: 0.7328767123287672 
# F1: 0.7430555555555556 
# MCC: 0.4933358710758228

9


# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


train_Feature = np.load("../../data_all/TCRA_train_feature_array.npy")    
train_Label = np.load("../../data_all/TCRA_train_label_array.npy")

test_Feature = np.load("../../data_all/TCRA_test_feature_array.npy")    
test_Label = np.load("../../data_all/TCRA_test_label_array.npy")

X_train = train_Feature
Y_train = train_Label#[:,1]  

print(X_train.shape)

X_test = test_Feature
Y_test = test_Label#[:,1]  

print(X_test.shape)


# In[ ]:




