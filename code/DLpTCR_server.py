#!/usr/bin/env python
# coding: utf-8

# # 1、导入库




import tensorflow.compat.v1 as tf

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

from sklearn.utils import shuffle

from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt
import csv
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示error和warining信息 3 只显示error信息
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 这一行注释掉就是使用cpu，不注释就是使用gpu

# # 2、TCRA预测函数


def TCRA_Model_Integration(FULL_M,CNN_M,RESNET_M,FULL_Feature,CNN_Feature,RESNET_Feature):
    K.clear_session()
    tf.reset_default_graph()
    
    FULL_model = load_model(FULL_M)
    CNN_model = load_model(CNN_M)
    RESNET_model = load_model(RESNET_M)
    
    
    FULL_X = FULL_Feature
    FULL_X = FULL_X.reshape([len(FULL_X),29,20,1])

    CNN_X = CNN_Feature
    CNN_X = CNN_X.reshape([len(CNN_X),29,20,1])

    RESNET_X = RESNET_Feature
    RESNET_X = RESNET_X.reshape([len(RESNET_X),20,16,2])

    Y_PRED_FULL = FULL_model.predict(FULL_X)
    Y_PRED_CNN = CNN_model.predict(CNN_X)
    Y_PRED_RESNET = RESNET_model.predict(RESNET_X)


    Y_pred_FULL = np.argmax(Y_PRED_FULL, axis=-1)
    Y_pred_CNN = np.argmax(Y_PRED_CNN, axis=-1)
    Y_pred_RESNET = np.argmin(Y_PRED_RESNET, axis=-1)


    Y_pred_ALL_avg = np.zeros([len(Y_pred_FULL),2])
    Y_pred_ALL = np.zeros([len(Y_pred_FULL),2])
          
            
            
    for i in range(len(Y_PRED_FULL)):
        Y_pred_ALL_avg[i,0] = (Y_PRED_FULL[i,1]+Y_PRED_CNN[i,1]+Y_PRED_RESNET[i,0])/3
        
        Y_pred_ALL_avg[i,1] = (Y_PRED_FULL[i,0]+Y_PRED_CNN[i,0]+Y_PRED_RESNET[i,1])/3
        
        if Y_pred_ALL_avg[i,0]>0.5:
            Y_pred_ALL[i,0] = 1
            Y_pred_ALL[i,1] = 0
        else:
            Y_pred_ALL[i,0] = 0
            Y_pred_ALL[i,1] = 1
    
    del FULL_model
    del CNN_model
    del RESNET_model

    return Y_pred_ALL,Y_pred_ALL_avg


# # 3、TCRB预测函数

def TCRB_Model_Integration(FULL_M,CNN_M,RESNET_M,FULL_Feature,CNN_Feature,RESNET_Feature):
    K.clear_session()
    tf.reset_default_graph()   
    
    FULL_model = load_model(FULL_M)
    CNN_model = load_model(CNN_M)
    RESNET_model = load_model(RESNET_M) 
    
    FULL_X = FULL_Feature
    FULL_X = FULL_X.reshape([len(FULL_X),20,19,2])

    CNN_X = CNN_Feature
    CNN_X = CNN_X.reshape([len(CNN_X),20,21,2])  
    
    RESNET_X = RESNET_Feature
    RESNET_X = RESNET_X.reshape([len(RESNET_X),20,11,2])

    Y_PRED_FULL = FULL_model.predict(FULL_X)
    Y_PRED_CNN = CNN_model.predict(CNN_X)
    Y_PRED_RESNET = RESNET_model.predict(RESNET_X)

    Y_pred_FULL = np.argmin(Y_PRED_FULL, axis=-1)
    Y_pred_CNN = np.argmin(Y_PRED_CNN, axis=-1)
    Y_pred_RESNET = np.argmin(Y_PRED_RESNET, axis=-1)


    Y_pred_ALL_avg = np.zeros([len(Y_pred_FULL),2])
    Y_pred_ALL = np.zeros([len(Y_pred_FULL),2])
          
            
            
    for i in range(len(Y_PRED_FULL)):
        Y_pred_ALL_avg[i,0] = (Y_PRED_FULL[i,0]+Y_PRED_CNN[i,0]+Y_PRED_RESNET[i,0])/3
        
        Y_pred_ALL_avg[i,1] = (Y_PRED_FULL[i,1]+Y_PRED_CNN[i,1]+Y_PRED_RESNET[i,1])/3
        
        if Y_pred_ALL_avg[i,0]>0.5:
            Y_pred_ALL[i,0] = 1
            Y_pred_ALL[i,1] = 0
        else:
            Y_pred_ALL[i,0] = 0
            Y_pred_ALL[i,1] = 1
            
            
    del FULL_model
    del CNN_model
    del RESNET_model
 
    return Y_pred_ALL,Y_pred_ALL_avg
    


# # 4、TCA结果预测(预测为0表示该CDR3与其Epitope结合)

def pred_A(user_dir):
    TCRA_FULL_M = './model/FULL_A_ALL_onehot.h5'
    TCRA_CNN_M = './model/CNN_A_ALL_onehot.h5'
    TCRA_RESNET_M = './model/RESNET_A_ALL_pca15.h5'
    
    

    TCRA_FULL_Feature = np.load(str(user_dir) + "/TCRA_onehot_feature_array.npy") 
    TCRA_FULL_Feature = TCRA_FULL_Feature[:,0:29,:]
    TCRA_CNN_Feature= np.load(str(user_dir) + "/TCRA_onehot_feature_array.npy")   
    TCRA_CNN_Feature = TCRA_CNN_Feature[:,0:29,:]    
    TCRA_RESNET_Feature = np.load(str(user_dir) + "/TCRA_PCA15_feature_array.npy") 

    TCRA_Y_pred_ALL,TCRA_Y_pred_ALL_avg = TCRA_Model_Integration(TCRA_FULL_M,TCRA_CNN_M,TCRA_RESNET_M,
                                             TCRA_FULL_Feature,TCRA_CNN_Feature,TCRA_RESNET_Feature)
    return TCRA_Y_pred_ALL,TCRA_Y_pred_ALL_avg


# # 5、TCRB结果预测(预测为0表示该TCRB与其Epitope结合)

def pred_B(user_dir):
    TCRB_FULL_M = './model/FULL_B_ALL_pca18.h5'
    TCRB_CNN_M = './model/CNN_B_ALL_pca20.h5'
    TCRB_RESNET_M = './model/RESNET_B_ALL_pca10.h5'

    TCRB_FULL_Feature = np.load(str(user_dir) + "/TCRB_PCA18_feature_array.npy")
    TCRB_CNN_Feature= np.load(str(user_dir) + "/TCRB_PCA20_feature_array.npy")    
    TCRB_RESNET_Feature = np.load(str(user_dir) + "/TCRB_PCA10_feature_array.npy") 
    
    TCRB_Y_pred_ALL,TCRB_Y_pred_ALL_avg  = TCRB_Model_Integration(TCRB_FULL_M,TCRB_CNN_M,TCRB_RESNET_M,
                                              TCRB_FULL_Feature,TCRB_CNN_Feature,TCRB_RESNET_Feature)
    
    return TCRB_Y_pred_ALL,TCRB_Y_pred_ALL_avg


# # 6、结果整合，只有TCRA和TCRB同时与其Epitope结合才判定为正样本

def pred_inte_all(user_dir,user_select):
    
    
    if user_select == 'A':
        TCRA_Y_pred_ALL,TCRA_Y_pred_ALL_avg = pred_A(user_dir)
        return TCRA_Y_pred_ALL,TCRA_Y_pred_ALL_avg
        
    elif user_select == 'B':
        TCRB_Y_pred_ALL,TCRB_Y_pred_ALL_avg = pred_B(user_dir)
        return TCRB_Y_pred_ALL,TCRB_Y_pred_ALL_avg
        
    elif user_select == 'AB':    
        TCRA_Y_pred_ALL,TCRA_Y_pred_ALL_avg = pred_A(user_dir)
        TCRB_Y_pred_ALL,TCRB_Y_pred_ALL_avg = pred_B(user_dir)
        
        TCRAB_acc_ALL = [list(TCRA_Y_pred_ALL_avg[:,0]),list(TCRB_Y_pred_ALL_avg[:,0])]

        Y_pred_ALL = np.zeros(len(TCRA_Y_pred_ALL),)
        for i in range(len(TCRA_Y_pred_ALL)):
            if TCRA_Y_pred_ALL[i,0]+TCRB_Y_pred_ALL[i,0] > 0:  
                Y_pred_ALL[i] = 1
                
                
                
        return Y_pred_ALL,TCRAB_acc_ALL



def save_outputfile(user_dir,user_select,excel_file_path,TCRA_cdr3,TCRB_cdr3,Epitope):
    
    

    

    if user_select == 'A':
        print(user_select)

        TCRA_Y_pred_ALL,TCRA_Y_pred_ALL_avg = pred_inte_all(user_dir,user_select)  
        #print(TCRA_Y_pred_ALL)
        print('1')
        
        n_TCRA_Y_pred_ALL= list(range(len(TCRA_Y_pred_ALL)))
        n_TCRA_Y_acc_ALL= list(range(len(TCRA_Y_pred_ALL)))

        for k in range(len(TCRA_Y_pred_ALL)):
            
            if TCRA_Y_pred_ALL[k,0] == 0:
                n_TCRA_Y_pred_ALL[k] = 'True TCR-pMHC'
                n_TCRA_Y_acc_ALL[k] = 1 - TCRA_Y_pred_ALL_avg[k,0]
                
            if TCRA_Y_pred_ALL[k,0] == 1:
                n_TCRA_Y_pred_ALL[k] = 'False TCR-pMHC'
                n_TCRA_Y_acc_ALL[k] = 1 - TCRA_Y_pred_ALL_avg[k,0]                

        #字典中的key值即为csv中列名
        dataframe = pd.DataFrame({'TCRA_CDR3':TCRA_cdr3,'Epitope':Epitope,'Predict':n_TCRA_Y_pred_ALL,'Probability (predicted as a positive sample)':n_TCRA_Y_acc_ALL})

        #将DataFrame存储为csv,index表示是否显示行名，default=True
        print('2')
        csv_file = 'TCRA_pred.csv'
        csv_path = str(user_dir) + str(csv_file)
        dataframe.to_csv(csv_path,sep=',')
        print('3')

    elif user_select == 'B':
        print(user_select)

        TCRB_Y_pred_ALL,TCRB_Y_pred_ALL_avg=  pred_inte_all(user_dir,user_select)
        print('1')
        
        n_TCRB_Y_pred_ALL= list(range(len(TCRB_Y_pred_ALL)))
        
        n_TCRB_Y_acc_ALL = list(range(len(TCRB_Y_pred_ALL)))
        
        
        for k in range(len(TCRB_Y_pred_ALL)):
            
            if TCRB_Y_pred_ALL[k,0] == 0:
                n_TCRB_Y_pred_ALL[k] = 'True TCR-pMHC'
                n_TCRB_Y_acc_ALL[k] = 1 - TCRB_Y_pred_ALL_avg[k,0]
                
            if TCRB_Y_pred_ALL[k,0] == 1:
                n_TCRB_Y_pred_ALL[k] = 'False TCR-pMHC'
                n_TCRB_Y_acc_ALL[k] = 1 - TCRB_Y_pred_ALL_avg[k,0]
                
                
        print('2')
        #字典中的key值即为csv中列名
        dataframe = pd.DataFrame({'TCRB_CDR3':TCRB_cdr3,'Epitope':Epitope,'Predict':n_TCRB_Y_pred_ALL,'Probability (predicted as a positive sample)':n_TCRB_Y_acc_ALL})

        #将DataFrame存储为csv,index表示是否显示行名，default=True
        
        csv_file = 'TCRB_pred.csv'
        csv_path = str(user_dir) + str(csv_file)
        
        dataframe.to_csv(csv_path,sep=',')
        print('3')

        
    elif user_select == 'AB':
        print(user_select)
        
        TCRAB_Y_pred_ALL ,TCRAB_Y_acc_ALL=  pred_inte_all(user_dir,user_select)
        print('1')
        
        n_TCRAB_Y_pred_ALL= list(range(len(TCRAB_Y_pred_ALL)))
        
        n_TCRA_Y_acc_ALL= list(range(len(TCRAB_Y_pred_ALL)))
        n_TCRB_Y_acc_ALL= list(range(len(TCRAB_Y_pred_ALL)))
        
        for k in range(len(TCRAB_Y_pred_ALL)):
            if TCRAB_Y_pred_ALL[k] == 0:
                n_TCRAB_Y_pred_ALL[k] = 'True TCR-pMHC'
                
                n_TCRA_Y_acc_ALL[k] = 1 - TCRAB_Y_acc_ALL[0][k]
                n_TCRB_Y_acc_ALL[k] = 1 - TCRAB_Y_acc_ALL[1][k]               
                
            if TCRAB_Y_pred_ALL[k] == 1:
                n_TCRAB_Y_pred_ALL[k] = 'False TCR-pMHC'
                n_TCRA_Y_acc_ALL[k] = 1 - TCRAB_Y_acc_ALL[0][k]
                n_TCRB_Y_acc_ALL[k] = 1 - TCRAB_Y_acc_ALL[1][k] 
                
        print('2')
        #字典中的key值即为csv中列名
        dataframe = pd.DataFrame({'TCRA_CDR3':TCRA_cdr3,'TCRB_CDR3':TCRB_cdr3,'Epitope':Epitope,'Predict':n_TCRAB_Y_pred_ALL,
                                'Probability (TCRA_Epitope)':n_TCRA_Y_acc_ALL,'Probability (TCRB_Epitope)':n_TCRB_Y_acc_ALL})
        
        #将DataFrame存储为csv,index表示是否显示行名，default=True
        
        csv_file = 'TCRAB_pred.csv'
        csv_path = str(user_dir) + str(csv_file)
        dataframe.to_csv(csv_path,sep=',')
        print('3')
        
    return csv_file
      
          




