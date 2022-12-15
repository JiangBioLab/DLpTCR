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
from tensorflow.python.keras.callbacks import ReduceLROnPlateau,LearningRateScheduler, ModelCheckpoint
from keras.utils import plot_model

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.python.keras.models import load_model
import pandas
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import matplotlib.pyplot as plt
import csv
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示error和warining信息 3 只显示error信息
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 这一行注释掉就是使用cpu，不注释就是使用gpu



def RESNET_pca15(model_number,modelfile,Epochs= 20,Batch_size=32,PCA_num = 15):
    # 优化器选择 Adam 优化器。
    # 损失函数使用 sparse_categorical_crossentropy，
    # 还有一个损失函数是 categorical_crossentropy，两者的区别在于输入的真实标签的形式，
    # sparse_categorical 输入的是整形的标签，例如 [1, 2, 3, 4]，categorical 输入的是 one-hot 编码的标签。
    

    train_Feature = np.load("../../data/train_TCRA_PCA{}_feature_array.npy".format(PCA_num))    
    train_Label = np.load("../../data/train_TCRA_PCA{}_label_array.npy".format(PCA_num))
    
    test_Feature = np.load("../../data/test_TCRA_PCA{}_feature_array.npy".format(PCA_num)) 
    test_Label = np.load("../../data/test_TCRA_PCA{}_label_array.npy".format(PCA_num))
      
           
    X_train = train_Feature
    Y_train = train_Label#[:,1]  
           
    X_test = test_Feature
    Y_test = test_Label#[:,1]  
    
    
    X_train,Y_train = shuffle(X_train,Y_train)
    X_test,Y_test = shuffle(X_test,Y_test)

    

    modelfile = modelfile

    X_train= X_train.reshape([len(X_train),20,PCA_num+1,2])
    X_test = X_test.reshape([len(X_test),20,PCA_num+1,2])
    X_test=tf.cast(X_test, tf.float32)
    Epochs = Epochs
    Batch_size = Batch_size
    history1 = resnet_attention_train_predict(20, PCA_num,model_number,modelfile,2,X_train, Y_train,X_test, Y_test,Epochs,Batch_size)
    
    return history1
                     




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
    outputs = layers.Dense(num_classes, 
                           activation='softmax',
                           kernel_initializer='he_normal')(y)
    # Instantiate model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model   


def resnet_attention_train_predict(row, PCA_num, model_number,modelfile, m, x_train, y_train,x_test, y_test,Epochs,Batch_size,):
    
    y_train, y_train,x_test, y_test = x_train, y_train,x_test, y_test

    model = resnet_v1(input_shape=(row, PCA_num+1, m), depth=20, num_classes=2)
    
    model.compile(optimizer="Adam",
              loss=keras.losses.binary_crossentropy,
              metrics=['accuracy'])
        
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), 
                                   cooldown=0, 
                                   patience=5,
                                   min_lr=0.5e-6)

    checkpoint = ModelCheckpoint(filepath=modelfile, 
                                 monitor='val_loss',
                                 verbose=0, 
                                 save_best_only=True)#,save_weights_only=True)
    
    cbs = [checkpoint, lr_reducer, lr_scheduler]

    history = model.fit(x_train, y_train,
                        batch_size=Batch_size,
                        epochs=Epochs,
                        verbose=0, 
                        #validation_split=0.1,
                        validation_data=(x_test, y_test),
                        shuffle=False,
                        callbacks=cbs)#callbacks=cbs
    return history
    
    del model
    
 


def computing_result(Feature_array,Label_array,model):
    
    X_TEST = Feature_array
    Y_TEST = Label_array
    
    model1 = model
    Y_PRED = model1.predict(X_TEST)

    Y_pred2 = np.argmin(Y_PRED, axis=-1) #使用pca特征时  标签需用np.argmin 替换np.argmax
    Y_test2 = np.argmin(Y_TEST, axis=-1)
#     print(Y_pred2)
#     print(Y_test2)
    confusion_matrix1 =confusion_matrix(Y_test2,Y_pred2)
    
    new_confusion_matrix1 = [[confusion_matrix1[1,1],confusion_matrix1[1,0]],[confusion_matrix1[0,1],confusion_matrix1[0,0]]]
    accuracy = accuracy_score(Y_test2,Y_pred2) #准确率
    precision = precision_score(Y_test2,Y_pred2) #精确率
    recall = recall_score(Y_test2,Y_pred2) #召回率
    f1= f1_score(Y_test2,Y_pred2) #F1
    MCC = matthews_corrcoef(Y_test2,Y_pred2) #MCC



    fpr, tpr, thresholds = metrics.roc_curve(Y_TEST[:,1], Y_PRED[:,1])
    roc_auc = auc(fpr, tpr)
    
    del model1
    return new_confusion_matrix1,accuracy,precision,recall,f1,MCC,fpr,tpr,roc_auc

def roc_plot(fpr,tpr,roc_auc):
    #开始画ROC曲线
    plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.01,1.01])
    plt.ylim([0,1.01])
    plt.xlabel('False Positive Rate') #横坐标是fpr
    plt.ylabel('True Positive Rate')  #纵坐标是tpr
    plt.title('Receiver operating characteristic')
    plt.show()    
    
    
    




for model_number in range(1,51):
    modelfile = './model/RESNET_A_ALL_test_pca15_{}.h5'.format(model_number)
    RESNET_pca15(model_number,modelfile,8,8,15)




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

csvFile = open("RESNET_A_ALL_pca15_result1.csv", "w" , newline='')
csv_writer = csv.writer(csvFile)
csv_writer.writerow(fileHeader)
PCA_num = 15
for model_number in range(1,51):

    modelfile = './model/RESNET_A_ALL_pca15_{}.h5'.format(model_number)
    model = load_model(modelfile)


    test_Feature = np.load("../../data/test_TCRA_PCA{}_feature_array.npy".format(PCA_num)) 
    test_Label = np.load("../../data/test_TCRA_PCA{}_label_array.npy".format(PCA_num))
    


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






