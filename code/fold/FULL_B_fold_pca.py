

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

from tensorflow.python.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示error和warining信息 3 只显示error信息
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 这一行注释掉就是使用cpu，不注释就是使用gpu






#测试集
def FULL_pca(Dropout1 = 0, Epochs = 20, Batch_size = 64, PCA_num = 18 ):
    # 优化器选择 Adam 优化器。
    # 损失函数使用 sparse_categorical_crossentropy，
    # 还有一个损失函数是 categorical_crossentropy，两者的区别在于输入的真实标签的形式，
    # sparse_categorical 输入的是整形的标签，例如 [1, 2, 3, 4]，categorical 输入的是 one-hot 编码的标签。
    
    Feature_test = np.load("E:/yanyi/CDR3/process/net_resnet/data/train_TCRA_PCA{}_feature_array.npy".format(PCA_num))    
    Label_array = np.load("E:/yanyi/CDR3/process/net_resnet/data//train_TCRA_PCA{}_label_array.npy".format(PCA_num))
    print('\n\nPCA_NUM: {}'.format(PCA_num))
    print('Feature.shape: {}'.format(Feature_test.shape))
    #print('Label.shape: {}'.format(Label_array.shape))

       
    X = Feature_test#[:,29:58,:]
    Y = Label_array[:,1]

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
                
        X_train= X_train.reshape([len(X_train),20,PCA_num+1,2])
        X_test = X_test.reshape([len(X_test),20,PCA_num+1,2])
        X_test=tf.cast(X_test, tf.float32)

        model = tf.keras.models.Sequential([


            tf.keras.layers.Flatten(input_shape=(20,PCA_num+1,2)),

            tf.keras.layers.Dense(256,activation='relu'),# kernel_regularizer=regularizers.l2(0.01)),#  activation='relu',
            tf.keras.layers.Dense(512,activation='relu'),# kernel_regularizer=regularizers.l2(0.01)),#  activation='relu',

            tf.keras.layers.Dense(256,activation='relu'),# kernel_regularizer=regularizers.l2(0.01)),#  activation='relu',
            #tf.keras.layers.LeakyReLU(alpha=0.05), 



            tf.keras.layers.Dense(128,activation='relu'),
            #tf.keras.layers.LeakyReLU(alpha=0.05), 
            tf.keras.layers.Dense(64,activation='relu'),
            #tf.keras.layers.LeakyReLU(alpha=0.05), 
            tf.keras.layers.Dropout(Dropout1),# Dropout:在 0 和 1 之间浮动。需要丢弃的输入比例
            tf.keras.layers.Dense(1, activation='sigmoid')




        ]) 

        model.compile(optimizer="Adam",
                      loss=keras.losses.binary_crossentropy,
                      metrics=['accuracy'])
        model.fit(X_train, Y_train, epochs= Epochs , batch_size= Batch_size, verbose=0,)
        
        
        Y_pred = model.predict_classes(X_test)
        #print(Y_pred)
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
        #print(aa)
        
        
        if aa == 1:
            Y_test_all = Y_test
            Y_pred_all = Y_pred
        else:
            Y_test_all = np.append(Y_test_all, Y_test, axis=0)
            Y_pred_all = np.append(Y_pred_all, Y_pred, axis=0) 

        aa += 1
        del model
        
    print('总混淆矩阵')
    print(TP,FN)
    print(FP,TN)
    
    #print(Y_test_all[0])
            
    accuracy = accuracy_score(Y_test_all,Y_pred_all) #准确率
    precision = precision_score(Y_test_all,Y_pred_all) #精确率
    recall = recall_score(Y_test_all,Y_pred_all) #召回率
    f1= f1_score(Y_test_all,Y_pred_all) #F1
    
    MCC = matthews_corrcoef(Y_test_all,Y_pred_all) #MCC

   
    
    print('准确率ACC:',accuracy,
          '\n精确率precision:',precision,
          '\n召回率recall:',recall,
          '\nF1:',f1,
          '\nMCC:',MCC 
         )
    
    

for i in range(8,21):
    
    FULL_pca(0.3,50,128,i)

