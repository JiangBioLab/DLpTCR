
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
AminoAcids = 'ARNDCQEGHILKMFPSTWYV'
def aaindex1Values():
    AAValues=[]
    newlineFlag = False
    addValueFlag = False
    t = []
    with open('./aaindex1') as fr:
        lines = fr.readlines()
        for line in lines:
            if line.startswith('I'):
                newlineFlag = True
                addValueFlag = True
                t = []
                continue
            elif line.startswith('//'):
                newlineFlag = False
                if addValueFlag:
                    AAValues.append(t)
     
            if newlineFlag:
                line = line.strip()
                if 'NA' in line:
                    addValueFlag = False
                    continue
                vals = line.split()
                for v in vals:
                    t.append(eval(v))
    return np.array(AAValues)      

def aaindex1PCAValues(n_features=15):
    file = './Amino_Acids_PCAVal{}_dict.txt'.format(n_features)
    with open(file,'r') as fr:
        aadic = eval(fr.read())
    return aadic
        
if __name__ == '__main__':
    aavals = aaindex1Values().T
    
    scaler = StandardScaler()
    aa_scal = scaler.fit_transform(aavals)
    pca = PCA(n_components=19)
    aa_pca = pca.fit_transform(aa_scal)  
    aaval_dic = {}
    for i in range(20):
        aaval_dic[AminoAcids[i]] = list(aa_pca[i])
    with open('./Amino_Acids_PCAVal19_dict.txt','w') as fw:
        fw.write(str(aaval_dic))
    
        
    