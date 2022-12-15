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

from aaindexValues import aaindex1PCAValues


def pca_code(seqs:list, row=30, n_features=16):
    aadict = aaindex1PCAValues(n_features)
    x = []
    col = n_features+1
    for i in range(len(seqs)):
        seq = seqs[i]
        n = len(seq)
        t = np.zeros(shape=(row, col))
        j = 0
        while j < n and j < row:
            t[j,:-1] = aadict[seq[j]]
            t[j,-1] = 0
            j += 1
        while j < row:
            t[j,-1] = 1
            j = j + 1
        x.append(t)
    return np.array(x)

def read_seqs(file, model=1):
    data = pandas.read_csv(file)
    labels = data.Class_label
    cdr3 = data.CDR3
    epitope = data.Epitope
    cdr3_seqs, epit_seqs = [], []
    for i in range(len(epitope)):
        if model == 1:
            cdr3_seqs.append(cdr3[i][2:-1])
        elif model == 2:
            cdr3_seqs.append(cdr3[i])
        epit_seqs.append(epitope[i])
    
    return cdr3_seqs, epit_seqs, labels

def load_data(col=20, row=9, m=1):
    
    train_cdr3_seqs,  train_epit_seqs, train_labels = read_seqs(trainFile, m)
    x_train = np.ndarray(shape=(len(train_cdr3_seqs), row, col+1, 2)) #改变数据集的通道数量和shape大小
    x_train[:,:,:,0] = pca_code(train_cdr3_seqs, row, col) ##第一个通道
    x_train[:,:,:,1] = pca_code(train_epit_seqs, row, col) ##第二个通道
     
    y_train = np.array(train_labels)
    y_train = to_categorical(y_train, 2)
    
    test_cdr3_seqs,  test_epit_seqs, test_labels = read_seqs(testFile, m)
    x_test = np.ndarray(shape=(len(test_cdr3_seqs), row, col+1, 2))
    x_test[:,:,:,0] = pca_code(test_cdr3_seqs, row, col)
    x_test[:,:,:,1] = pca_code(test_epit_seqs, row, col)
    
    y_test = np.array(test_labels)
    y_test = to_categorical(y_test, 2)
    
    indt_cdr3_seqs,  indt_epit_seqs, indt_labels = read_seqs(indepFile, m)
    x_indt = np.ndarray(shape=(len(indt_cdr3_seqs), row, col+1, 2))
    x_indt[:,:,:,0] = pca_code(indt_cdr3_seqs, row, col)
    x_indt[:,:,:,1] = pca_code(indt_epit_seqs, row, col)
    
    y_indt = np.array(indt_labels)
    y_indt = to_categorical(y_indt, 2)

    
    return (x_train, y_train), (x_test, y_test),(x_indt, y_indt)



trainFile = '../data/TCRB_train.csv'
testFile = '../data/TCRB_test.csv'       ####
indepFile = '../data/TCRB_COVID-19.csv'  ####


m=2
row = 20
#col = 18  #PCA 降维后的特征数量

for i in range(8,21):
    col = i #PCA 降维后的特征数量
    (x_train, y_train), (x_test, y_test), (x_indt, y_indt) = load_data(col=col, row=row, m=m)



    np.save('../data/train_TCRB_PCA{}_feature_array'.format(col),x_train)
    np.save('../data/train_TCRB_PCA{}_label_array'.format(col),y_train)
    
    np.save('../data/test_TCRB_PCA{}_feature_array'.format(col),x_test)
    np.save('../data/test_TCRB_PCA{}_label_array'.format(col),y_test)





trainFile = '../data/TCRA_train.csv'
testFile = '../data/TCRA_test.csv'       ####
indepFile = '../data/TCRA_COVID-19.csv'  ####


m=2
row = 20
#col = 18  #PCA 降维后的特征数量

for i in range(8,21):
    col = i #PCA 降维后的特征数量
    (x_train, y_train), (x_test, y_test), (x_indt, y_indt) = load_data(col=col, row=row, m=m)



    np.save('../data/train_TCRA_PCA{}_feature_array'.format(col),x_train)
    np.save('../data/train_TCRA_PCA{}_label_array'.format(col),y_train)
    
    np.save('../data/test_TCRA_PCA{}_feature_array'.format(col),x_test)
    np.save('../data/test_TCRA_PCA{}_label_array'.format(col),y_test)




