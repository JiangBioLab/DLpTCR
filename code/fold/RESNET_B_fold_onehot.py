


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

import pandas
from sklearn.utils import shuffle

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint

import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示error和warining信息 3 只显示error信息
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 这一行注释掉就是使用cpu，不注释就是使用gpu



#测试集
def RESNET_onehot(Dropout1=0,Epochs= 20,Batch_size=64):
    # 优化器选择 Adam 优化器。
    # 损失函数使用 sparse_categorical_crossentropy，
    # 还有一个损失函数是 categorical_crossentropy，两者的区别在于输入的真实标签的形式，
    # sparse_categorical 输入的是整形的标签，例如 [1, 2, 3, 4]，categorical 输入的是 one-hot 编码的标签。
    
    Feature_test = np.load("../../data/TCRB_train_feature_array.npy")    
    Label_array = np.load("../../data/TCRB_train_label_array.npy")
       
    X = Feature_test[:,0:29,:] #提取one-hot特征
    #print(X[0])
    Y = Label_array#[:,1]

    X = X.reshape(len(X),-1)
    #loo = LeaveOneOut()
    
    kf = KFold(n_splits=5,shuffle=True,random_state=0)
    kf.get_n_splits(X)
    TN = FP = FN = TP = 0
    aa = 1 

    for train_index, test_index in kf.split(X):
        np.random.shuffle(train_index)
        np.random.shuffle(test_index)
        
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        
        
        
        
        
        X_train= X_train.reshape([len(X_train),29,20,1])
        X_test = X_test.reshape([len(X_test),29,20,1])
        X_test=tf.cast(X_test, tf.float32)
        info = "using the one_hoot th model\n"
        modelfile = './tra_resnet_model.h5'
        Y_pred = resnet_attention_train_predict(29, 19 ,modelfile,info,1,X_train, Y_train,X_test, Y_test)
        
        
        Y_pred = np.argmax(Y_pred, axis=-1)
        Y_test = np.argmax(Y_test, axis=-1)

        confusion_matrix1 =confusion_matrix(Y_test,Y_pred)
        
        
        
        
        
        TP += confusion_matrix1[0,0]
        FN += confusion_matrix1[0,1]
        FP += confusion_matrix1[1,0]
        TN += confusion_matrix1[1,1]
        
#         accuracy = accuracy_score(Y_test,Y_pred) #准确率
#         precision = precision_score(Y_test,Y_pred) #精确率
#         recall = recall_score(Y_test,Y_pred) #召回率
#         f1= f1_score(Y_test,Y_pred) #F1
               
#         print('混淆矩阵\n',confusion_matrix1,
#               '\n准确率ACC:',accuracy,
#               '\n精确率precision:',precision,
#               '\n召回率recall:',recall,
#               '\nF1:',f1,
#              )
               
#         y_predict = model.predict(X_test)      
        
#         y_probs = model.predict_proba(X_test) #模型的预测得分
#         #print(y_probs)
        
#         fpr, tpr, thresholds = metrics.roc_curve(Y_test,y_probs)
#         roc_auc = auc(fpr, tpr)  #auc为Roc曲线下的面积
#         #开始画ROC曲线
#         plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
#         plt.legend(loc='lower right')
#         plt.plot([0,1],[0,1],'r--')
#         plt.xlim([-0.1,1.1])
#         plt.ylim([-0.1,1.1])
#         plt.xlabel('False Positive Rate') #横坐标是fpr
#         plt.ylabel('True Positive Rate')  #纵坐标是tpr
#         plt.title('Receiver operating characteristic example')
#         plt.show()
        
        #model.save('./data_625/model_'+str(aa)+'.h5')
        print(aa)
        
        
        if aa == 1:
            Y_test_all = Y_test
            Y_pred_all = Y_pred
        else:
            Y_test_all = np.append(Y_test_all, Y_test, axis=0)
            Y_pred_all = np.append(Y_pred_all, Y_pred, axis=0) 

        aa += 1
        
        
    print('\n\n总混淆矩阵')
    print(TP,FN)
    print(FP,TN)
    
    #print(Y_test_all[0])
            
    accuracy = accuracy_score(Y_test_all,Y_pred_all) #准确率
    precision = precision_score(Y_test_all,Y_pred_all) #精确率
    recall = recall_score(Y_test_all,Y_pred_all) #召回率
    f1= f1_score(Y_test_all,Y_pred_all) #F1
    
    MCC = matthews_corrcoef(Y_test_all,Y_pred_all) #MCC

   
    
    print('\n准确率ACC:',accuracy,
          '\n精确率precision:',precision,
          '\n召回率recall:',recall,
          '\nF1:',f1,
          '\nMCC:',MCC 
         )
    
    

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
        
#     lr_scheduler = LearningRateScheduler(lr_schedule)
#     lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5,
#                                    min_lr=0.5e-6)

    checkpoint = ModelCheckpoint(filepath=modelfile, monitor='val_loss',
                                verbose=1, save_best_only=True,
                                save_weights_only=True)
    cbs = [checkpoint]#, lr_reducer, lr_scheduler]
    model.fit(x_train, y_train,
                  batch_size=32,
                  epochs=20,
                  #validation_split=0.1,
                  validation_data=[x_test, y_test],
                  shuffle=False,
                  callbacks=cbs)#callbacks=cbs
    
    model.load_weights(modelfile)  
    
    y_pred = model.predict(x_test)   
    del model
    
    return y_pred

    


RESNET_onehot(0.3,50,128)

