from Model_Predict_Feature_Extraction import *

from DLpTCR_server import *

input_file_path = '../data/Example_file.xlsx'

'''
Please refer to document 'Example_file.xlsx' for the format of the input file.
Column names are not allowed to change.
'''


model_select = "AB"  

'''
model:pTCRα    user_select = "A" 
model:pTCRβ    user_select = "B" 
model:pTCRαβ  user_select = "AB" 

'''

job_dir_name = 'test'
user_dir = './user/' + str(job_dir_name) + '/'

'''
The predicted files will be stored in the path "user_dir".
'''
user_dir_Exists = os.path.exists(user_dir)
if not user_dir_Exists: 
    os.makedirs(user_dir)
    
error_info,TCRA_cdr3,TCRB_cdr3,Epitope = deal_file(input_file_path, user_dir, model_select)
output_file_path = save_outputfile(user_dir, user_select, input_file_path,TCRA_cdr3,TCRB_cdr3,Epitope)
