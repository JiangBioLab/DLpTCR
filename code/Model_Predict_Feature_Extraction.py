import pandas as pd 
import os
import numpy as np
import csv
from numpy import *
import time
from sklearn.decomposition import PCA

AminoAcids = 'ARNDCQEGHILKMFPSTWYV'

def aaindex1PCAValues(n_features=15):
    file = './pca/Amino_Acids_PCAVal{}_dict.txt'.format(n_features)
    with open(file, 'r') as fr:
        aadic = eval(fr.read())
    return aadic


def pca_code(seqs: list, row=30, n_features=16):
    aadict = aaindex1PCAValues(n_features)
    x = []
    col = n_features + 1
    for i in range(len(seqs)):
        seq = seqs[i]
        n = len(seq)
        t = np.zeros(shape=(row, col))
        j = 0
        while j < n and j < row:
            t[j, :-1] = aadict[seq[j]]
            t[j, -1] = 0
            j += 1
        while j < row:
            t[j, -1] = 1
            j = j + 1
        x.append(t)
    return np.array(x)


def read_seqs(cdr3, epitope, model=1):
    cdr3_seqs, epit_seqs = [], []
    for i in range(len(epitope)):
        if model == 1:
            cdr3_seqs.append(cdr3[i][2:-1])
        elif model == 2:
            cdr3_seqs.append(cdr3[i])
        epit_seqs.append(epitope[i])

    return cdr3_seqs, epit_seqs


def load_data(CDR3, Epitope, col=20, row=9, m=1): 

    new_cdr3_seqs, new_epit_seqs = read_seqs(CDR3, Epitope, m)

    x_feature = np.ndarray(shape=(len(new_cdr3_seqs), row, col + 1, 2))  
    x_feature[:, :, :, 0] = pca_code(new_cdr3_seqs, row, col) 
    x_feature[:, :, :, 1] = pca_code(new_epit_seqs, row, col)  

    return x_feature


def AA_ONE_HOT(AA):
    one_hot_dict = {
        'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'C': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'D': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'E': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'F': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'G': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'H': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'I': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'K': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'M': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'N': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'P': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'S': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        'T': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        'V': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        'W': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        'Y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }

    coding_arr = np.zeros((len(AA), 20), dtype=float)

    for i in range(len(AA)):
        coding_arr[i] = one_hot_dict[AA[i]]

    return coding_arr


def AA_CHEM(AA):
    AA_CHEM_dict = {
        'A': [-0.591, -1.302, -0.733, 1.57, -0.146, 0.62, -0.5, 15, 2.35, 9.87, 6.11, -1.338, -3.102, 0.52, 1.18, 4.349,
              -0.368, 0.36, 0.67, -9.475],
        'C': [-1.343, 0.465, -0.862, -1.02, -0.255, 0.29, -1, 47, 1.71, 10.78, 5.02, -1.511, 0.957, 1.12, 1.89, 4.686,
              4.53, 0.70, 0.38, -12.210],
        'D': [1.05, 0.302, -3.656, -0.259, -3.242, -0.9, 3, 59, 1.88, 9.6, 2.98, -0.204, 0.424, 0.77, 0.05, 4.765, 2.06,
              -1.09, -1.2, -12.144],
        'E': [1.357, -1.453, 1.477, 0.113, -0.837, -0.74, 3, 73, 2.19, 9.67, 3.08, -0.365, 2.009, 0.76, 0.11, 4.295,
              1.77, -0.83, -0.76, -13.815],
        'F': [-1.006, -0.59, 1.891, -0.397, 0.412, 1.19, -2.5, 91, 2.58, 9.24, 5.91, 2.877, -0.466, 0.86, 1.96, 4.663,
              1.06, 1.01, 2.3, -20.504],
        'G': [-0.384, 1.652, 1.33, 1.045, 2.064, 0.48, 0, 1, 2.34, 9.6, 6.06, -1.097, -2.746, 0.56, 0.49, 3.972, -0.525,
              -0.82, 0, -7.592],
        'H': [0.336, -0.417, -1.673, -1.474, -0.078, -0.4, -0.5, 82, 1.78, 8.97, 7.64, 2.269, -0.223, 0.94, 0.31, 4.630,
              0, 0.16, 0.64, -17.550],
        'I': [-1.239, -0.547, 2.131, 0.393, 0.816, 1.38, -1.8, 57, 2.32, 9.76, 6.04, -1.741, 0.424, 0.65, 1.45, 4.224,
              0.791, 2.17, 1.9, -15.608],
        'K': [1.831, -0.561, 0.533, -0.277, 1.648, -1.5, 3, 73, 2.2, 8.9, 9.47, -1.822, 3.95, 0.81, 0.06, 4.358, 0,
              -0.56, -0.57, -12.366],
        'L': [-1.019, -0.987, -1.505, 1.266, -0.912, 1.06, -1.8, 57, 2.36, 9.6, 6.04, -1.741, 0.424, 0.58, 3.23, 4.385,
              1.07, 1.18, 1.9, -15.728],
        'M': [-0.663, -1.524, 2.219, -1.005, 1.212, 0.64, -1.3, 75, 2.28, 9.21, 5.74, -1.741, 2.484, 1.25, 2.67, 4.513,
              0.656, 1.21, 2.4, -15.704],
        'N': [0.945, 0.828, 1.299, -0.169, 0.933, -0.78, 0.2, 58, 2.18, 9.09, 10.76, -0.204, 0.424, 0.79, 0.23, 4.755,
              0, -0.9, -0.6, -12.480],
        'P': [0.189, 2.081, -1.628, 0.421, -1.392, 0.12, 0, 42, 1.99, 10.6, 6.3, 1.979, -2.404, 0.61, 0.76, 4.471,
              -2.24, -0.06, 1.2, -11.893],
        'Q': [0.931, -0.179, -3.005, -0.503, -1.853, -0.85, 0.2, 72, 2.17, 9.13, 5.65, -0.365, 2.009, 0.86, 0.72, 4.373,
              0.731, -1.05, -0.22, -13.689],
        'R': [1.538, -0.055, 1.502, 0.44, 2.897, -2.53, 3, 101, 2.18, 9.09, 10.76, 1.169, 3.06, 0.6, 0.20, 4.396, -1.03,
              -0.52, -2.1, -16.225],
        'S': [-0.228, 1.399, -4.76, 0.67, -2.647, -0.18, 0.3, 31, 2.21, 9.15, 5.68, -1.511, 0.957, 0.64, 0.97, 4.498,
              -0.524, -0.6, 0.01, -10.518],
        'T': [-0.032, 0.326, 2.213, 0.908, 1.313, -0.05, -0.4, 45, 2.15, 9.12, 5.6, -1.641, -1.339, 0.56, 0.84, 4.346,
              0, -1.20, 0.52, -12.369],
        'V': [-1.337, -0.279, -0.544, 1.242, -1.262, 1.08, -1.5, 43, 2.29, 9.74, 6.02, -1.641, -1.339, 0.54, 1.08,
              4.184, 0.401, 1.21, 1.5, -13.867],
        'W': [-0.595, 0.009, 0.672, -2.128, -0.184, 0.81, -3.4, 130, 2.38, 9.39, 5.88, 5.913, -1, 1.82, 0.77, 4.702,
              1.60, 1.31, 2.6, -26.166],
        'Y': [0.26, 0.83, 3.097, -0.838, 1.512, 0.26, -2.3, 107, 2.2, 9.11, 5.63, 2.714, -0.672, 0.98, 0.39, 4.604,
              4.91, 1.05, 1.6, -20.232],
        'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }

    coding_arr = np.zeros((len(AA), 20), dtype=float)

    for i in range(len(AA)):
        coding_arr[i] = AA_CHEM_dict[AA[i]]

    return coding_arr


def seq2feature(cdr3, epitope):
    feature_array = np.zeros([len(cdr3), 58, 20])

    for i in range(len(cdr3)):
        cdr3_1 = cdr3[i].upper() 
        epitope_1 = epitope[i].upper() 
        cdr3_epitope_splice = cdr3_1 + epitope_1
        # print(cdr3_epitope_splice)
        new_cdr3_epitope_splice = cdr3_epitope_splice

        if len(cdr3_epitope_splice) != 29:
            for j in range(29 - len(cdr3_epitope_splice)):
                new_cdr3_epitope_splice = 'X' + new_cdr3_epitope_splice

        aa_onehot = AA_ONE_HOT(new_cdr3_epitope_splice)
        aa_chen = AA_CHEM(new_cdr3_epitope_splice)

        data = np.append(aa_onehot, aa_chen)  
        # print(data)
        dima = aa_onehot.shape  
        dimn = aa_chen.shape  

        cdr3_epitope = data.reshape(dima[0] + dimn[0], dima[1]) 

        feature_array[i] = cdr3_epitope
    return feature_array


def deal_file(excel_file_path, user_dir, user_select):
    error_info = 0
    #print(1)


    input_file = pd.read_excel(excel_file_path)

    input_count = input_file.count()
    
    index_num = input_count[0]
    TCRA_cdr3_num = input_count[1]
    TCRB_cdr3_num = input_count[2]
    Epitope_num = input_count[3]
    
    
    TCRA_cdr3 = input_file.TCRA_CDR3

    TCRB_cdr3 = input_file.TCRB_CDR3

    Epitope = input_file.Epitope

    full_TCRA_cdr3 = TCRA_cdr3[0:(TCRA_cdr3_num-0)]

    full_TCRB_cdr3 = TCRB_cdr3[0:(TCRB_cdr3_num-0)]

    full_Epitope = Epitope[0:(Epitope_num-0)]
    
    
    M = 2
    Row = 20
    #print(2)


    

    if user_select == 'A':
    
        if TCRA_cdr3_num == Epitope_num :

            #print('3.1')
            
           

            TCRA_onehot_feature_array = seq2feature(full_TCRA_cdr3, full_Epitope)
            np.save(user_dir + 'TCRA_onehot_feature_array', TCRA_onehot_feature_array)
            #print('3.2')
            TCRA_Col = 15  
            TCRA_pca_feature = load_data(full_TCRA_cdr3, full_Epitope, TCRA_Col, Row, M)
            np.save(user_dir + 'TCRA_PCA{}_feature_array'.format(TCRA_Col), TCRA_pca_feature)
            #print('3.3')
        else:
            
            error_info = 1
        

    elif user_select == 'B':

        #print('4')
        if TCRB_cdr3_num == Epitope_num :
            TCRB_Col = [10, 18, 20]
            for i in TCRB_Col:
                #print(i)
                TCRB_pca_feature = load_data(full_TCRB_cdr3, full_Epitope, i, Row, M)
                np.save(user_dir + 'TCRB_PCA{}_feature_array'.format(i), TCRB_pca_feature)
        else:
            
            error_info = 2        
            

    elif user_select == 'AB':

        if TCRA_cdr3_num == TCRB_cdr3_num == Epitope_num :

            #print('5.1')
            TCRA_onehot_feature_array = seq2feature(full_TCRA_cdr3, full_Epitope)
            np.save(user_dir + 'TCRA_onehot_feature_array', TCRA_onehot_feature_array)

            #print('5.2')
            TCRA_Col = 15  
            TCRA_pca_feature = load_data(full_TCRA_cdr3, full_Epitope, TCRA_Col, Row, M)
            np.save(user_dir + 'TCRA_PCA{}_feature_array'.format(TCRA_Col), TCRA_pca_feature)
            #print('5.3')

            TCRB_Col = [10, 18, 20]
            for i in TCRB_Col:
                #print(i)
                TCRB_pca_feature = load_data(full_TCRB_cdr3, full_Epitope, i, Row, M)
                np.save(user_dir + 'TCRB_PCA{}_feature_array'.format(i), TCRB_pca_feature)



        else:
            #print('6')

            
            error_info = 3


    return error_info,full_TCRA_cdr3,full_TCRB_cdr3,full_Epitope