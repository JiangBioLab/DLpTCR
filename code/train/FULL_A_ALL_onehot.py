

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
from keras.utils import plot_model



def FULL_onehot(modelfile,Dropout1=0,Epochs= 20,Batch_size=64,):
    # 优化器选择 Adam 优化器。
    # 损失函数使用 sparse_categorical_crossentropy，
    # 还有一个损失函数是 categorical_crossentropy，两者的区别在于输入的真实标签的形式，
    # sparse_categorical 输入的是整形的标签，例如 [1, 2, 3, 4]，categorical 输入的是 one-hot 编码的标签。
    
    train_Feature = np.load("../../data_all/TCRA_train_feature_array.npy")    
    train_Label = np.load("../../data_all/TCRA_train_label_array.npy")
    
    test_Feature = np.load("../../data_all/TCRA_test_feature_array.npy")    
    test_Label = np.load("../../data_all/TCRA_test_label_array.npy")
           
    X_train = train_Feature[:,0:29,:] 
    Y_train = train_Label 
           
    X_test = test_Feature[:,0:29,:] 
    Y_test = test_Label
  
    X_train,Y_train = shuffle(X_train,Y_train)
    X_test,Y_test = shuffle(X_test,Y_test)

    X_train= X_train.reshape([len(X_train),29,20,1])
    X_test = X_test.reshape([len(X_test),29,20,1])
    X_test=tf.cast(X_test, tf.float32)

    model = tf.keras.models.Sequential([


        tf.keras.layers.Flatten(input_shape=(29,20,1)),

        tf.keras.layers.Dense(256,activation='relu'),# kernel_regularizer=regularizers.l2(0.01)),#  activation='relu',
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
                  metrics=['accuracy'],
                  #lr=0.2
                 )   
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
                        validation_data=[X_test, Y_test],
                        #validation_split=0.5,
                        shuffle=True,
                        callbacks=cbs)
    
    
    #plot_model(model, to_file='model.png')
    return history

    
    
    del model




csvFile = open("FULL_A_ALL_onehot_ACC.csv", "w" , newline='')
csv_writer= csv.writer(csvFile)




for model_number in range(50):
    
    modelfile = './model/FULL_A_ALL_onehot_plt_{}.h5'.format(model_number)
    history = FULL_onehot(modelfile,0.3,300,128)


    test_row = history.history['val_accuracy']
    
    csv_writer.writerow(test_row)
    
csvFile.close() 





def computing_result(Feature_array,Label_array,model):
    
    X_TEST = Feature_array
    Y_TEST = Label_array
    
    model1 = model
    Y_PRED = model1.predict(X_TEST)

    Y_pred2 = np.argmax(Y_PRED, axis=-1)
    Y_test2 = np.argmax(Y_TEST, axis=-1)


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

csvFile = open("FULL_A_ALL_onehot_result.csv", "w" , newline='')
csv_writer = csv.writer(csvFile)
csv_writer.writerow(fileHeader)

for model_number in range(1,51):

    modelfile = './model/FULL_A_ALL_onehot_{}.h5'.format(model_number)
    model = load_model(modelfile)


    
    test_Feature = np.load("../../data_all/TCRA_test_feature_array.npy")    
    test_Label = np.load("../../data_all/TCRA_test_label_array.npy")

    X_test = test_Feature[:,0:29,:] 
    Y_test = test_Label
    X_test = X_test.reshape([len(X_test),29,20,1])
    
    test_CM,accuracy1,precision1,recall1,f11,MCC1,fpr1,tpr1,roc_auc1 = computing_result(X_test,Y_test,model)
    
    
    
    Feature_test2 = np.load("../../data_all/SARS-CoV-2_TCRA_feature_array.npy")    
    Label_array2 = np.load("../../data_all/SARS-CoV-2_TCRA_label_array.npy")

    X_SARS = Feature_test2[:,0:29,:] 

    Y_SARS = Label_array2
    X_SARS = X_SARS.reshape([len(X_SARS),29,20,1])


    SARS_CM,accuracy2,precision2,recall2,f12,MCC2,fpr2,tpr2,roc_auc2 = computing_result(X_SARS,Y_SARS,model)
    
    test_row = [model_number,'TEST',
                test_CM[0][0],test_CM[0][1],
                test_CM[1][0],test_CM[1][1],
                accuracy1,precision1,recall1,f11,MCC1,roc_auc1]
    
    
    SARS_CoV_2_row = [model_number,'SARS-CoV-2',
                      SARS_CM[0][0],SARS_CM[0][1],
                      SARS_CM[1][0],SARS_CM[1][1],
                      accuracy2,precision2,recall2,f12,MCC2,roc_auc2]
    
    csv_writer.writerow(test_row)
    csv_writer.writerow(SARS_CoV_2_row)
    
    del model
csvFile.close() 




