#!/usr/bin/env python
# coding: utf-8



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
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils import plot_model
#from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint



import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from tensorflow.python.keras.models import load_model


import matplotlib.pyplot as plt

import csv
import pandas as pd



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示error和warining信息 3 只显示error信息
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 这一行注释掉就是使用cpu，不注释就是使用gpu



def CNN_pca20(modelfile,Dropout1=0,Epochs= 20,Batch_size=64,PCA_num = 20):
    # 优化器选择 Adam 优化器。
    # 损失函数使用 sparse_categorical_crossentropy，
    # 还有一个损失函数是 categorical_crossentropy，两者的区别在于输入的真实标签的形式，
    # sparse_categorical 输入的是整形的标签，例如 [1, 2, 3, 4]，categorical 输入的是 one-hot 编码的标签。
    

    train_Feature = np.load("../../data/train_TCRB_PCA{}_feature_array.npy".format(PCA_num))    
    train_Label = np.load("../../data/train_TCRB_PCA{}_label_array.npy".format(PCA_num))
    
    test_Feature = np.load("../../data/test_TCRB_PCA{}_feature_array.npy".format(PCA_num)) 
    test_Label = np.load("../../data/test_TCRB_PCA{}_label_array.npy".format(PCA_num))
           
    X_train = train_Feature
    Y_train = train_Label 
           
    X_test = test_Feature
    Y_test = test_Label
  
    X_train,Y_train = shuffle(X_train,Y_train)
    X_test,Y_test = shuffle(X_test,Y_test)

    X_train= X_train.reshape([len(X_train),20,PCA_num+1,2])
    X_test = X_test.reshape([len(X_test),20,PCA_num+1,2])
    X_test=tf.cast(X_test, tf.float32)

    model = tf.keras.models.Sequential([
#             tf.keras.layers.Conv2D(16, (7,7),padding = 'same', input_shape=(29,20,2),activation='relu'),            
#             #tf.keras.layers.LeakyReLU(alpha=0.05),
#             tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (5,5),padding = 'same', input_shape=(20,PCA_num+1,2),activation='relu'),            
        #tf.keras.layers.LeakyReLU(alpha=0.05),
        #tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.AveragePooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3),padding = 'same',activation='relu'),
        #tf.keras.layers.LeakyReLU(alpha=0.05),            
        #tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.AveragePooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation='relu'),# kernel_regularizer=regularizers.l2(0.01)),#  activation='relu',
        tf.keras.layers.Dense(256,activation='relu'),# kernel_regularizer=regularizers.l2(0.01)),#  activation='relu',
        #tf.keras.layers.LeakyReLU(alpha=0.05), 
        tf.keras.layers.Dense(128,activation='relu'),
        #tf.keras.layers.LeakyReLU(alpha=0.05), 
        tf.keras.layers.Dense(64,activation='relu'),
        #tf.keras.layers.LeakyReLU(alpha=0.05), 
        tf.keras.layers.Dropout(Dropout1),# Dropout:在 0 和 1 之间浮动。需要丢弃的输入比例
        tf.keras.layers.Dense(2, activation='softmax')
    ]) 

    model.compile(optimizer="Adam",
                  loss=keras.losses.binary_crossentropy,
                  metrics=['accuracy'])   
    checkpoint = ModelCheckpoint(filepath=modelfile, 
                                 monitor='val_loss',
                                 verbose=0, 
                                 save_best_only=True)#,save_weights_only=True)
    cbs = [checkpoint]#, lr_reducer, lr_scheduler]
    history = model.fit(X_train, 
                        Y_train, 
                        epochs= Epochs , 
                        batch_size= Batch_size, 
                        verbose=0,
                        validation_data=(x_test, y_test),
                        shuffle=False,
                        callbacks=cbs)
    return history
    del model





csvFile = open("CNN_B_ALL_pca20_test_ACC.csv", "w" , newline='')
csv_writer= csv.writer(csvFile)




for model_number in range(1,51):
    print(model_number)
    modelfile = './model/CNN_B_ALL_pca20_plt_{}.h5'.format(model_number)
    history = CNN_pca20(modelfile,0.3,300,128,20)


    test_row = history.history['val_accuracy']
    
    csv_writer.writerow(test_row)
csvFile.close() 






def computing_result(Feature_array,Label_array,model):
    
    X_TEST = Feature_array
    Y_TEST = Label_array
    
    model1 = model
    Y_PRED = model1.predict(X_TEST)

    Y_pred2 = np.argmin(Y_PRED, axis=-1)
    Y_test2 = np.argmin(Y_TEST, axis=-1)


    confusion_matrix1 =confusion_matrix(Y_test2,Y_pred2)
    
    new_confusion_matrix1 = [[confusion_matrix1[1,1],confusion_matrix1[1,0]],[confusion_matrix1[0,1],confusion_matrix1[0,0]]]
    accuracy = accuracy_score(Y_test2,Y_pred2) #准确率
    precision = precision_score(Y_test2,Y_pred2) #精确率
    recall = recall_score(Y_test2,Y_pred2) #召回率
    f1= f1_score(Y_test2,Y_pred2) #F1
    MCC = matthews_corrcoef(Y_test2,Y_pred2) #MCC



    fpr, tpr, thresholds = metrics.roc_curve(Y_TEST[:,1], Y_PRED[:,1])
    roc_auc = auc(fpr, tpr)
    
    return new_confusion_matrix1,accuracy,precision,recall,f1,MCC,fpr,tpr,roc_auc

def roc_plot(fpr,tpr,roc_auc):
    #开始画ROC曲线
    plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.01,1.01])
    plt.ylim([0,1])
    plt.xlabel('False Positive Rate') #横坐标是fpr
    plt.ylabel('True Positive Rate')  #纵坐标是tpr
    plt.title('Receiver operating characteristic')
    plt.show()    
    
    
    



def writeMetrics(metricsFile,new_confusion_matrix1,accuracy,precision,recall,f1,MCC,roc_auc,noteInfo=''):
  
    with open(metricsFile,'a') as fw:
        if noteInfo:
            fw.write('\n\n' + noteInfo + '\n')
        fw.write('混淆矩阵\n',new_confusion_matrix1[0],'\n',new_confusion_matrix1[1])
        fw.write('\n准确率ACC:: %f '%accuracy)
        fw.write('\n精确率precision: %f '%precision)
        fw.write('\n召回率recall: %f '%recall)
        fw.write('\nF1: %f '%f1)
        fw.write('\nMCC: %f '%MCC)
        fw.write('\nAUC: %f '%roc_auc)
        



fileHeader =['model_number','dataset','TP','FN','FP','TN','ACC','precision','recall','f1','MCC','AUC']
# 写入数据

csvFile = open("CNN_B_ALL_pca20_result50.csv", "w" , newline='')
csv_writer = csv.writer(csvFile)
csv_writer.writerow(fileHeader)
PCA_num = 20
for model_number in range(1,51):

    modelfile = './model/CNN_B_ALL_pca20_{}.h5'.format(model_number)
    model = load_model(modelfile)


    test_Feature = np.load("../../data/test_TCRB_PCA{}_feature_array.npy".format(PCA_num)) 
    test_Label = np.load("../../data/test_TCRB_PCA{}_label_array.npy".format(PCA_num))
    


    X_test = test_Feature
    Y_test = test_Label
    X_test = X_test.reshape([len(X_test),20,PCA_num+1,2])
    
    test_CM,accuracy1,precision1,recall1,f11,MCC1,fpr1,tpr1,roc_auc1 = computing_result(X_test,Y_test,model)

    
    test_row = [model_number,'TEST',
                test_CM[0][0],test_CM[0][1],
                test_CM[1][0],test_CM[1][1],
                accuracy1,precision1,recall1,f11,MCC1,roc_auc1]
    

    
    csv_writer.writerow(test_row)

    
    del model
csvFile.close() 



