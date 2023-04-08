import os
import pickle
import numpy as np
import pandas as pd

S645_data = pd.read_excel('../PPI.xlsx', sheet_name='S645')
# S1131_data = pd.read_excel('../PPI.xlsx', sheet_name='S1131')
pdbs = S645_data['protein'].drop_duplicates().to_list()

# define the max seq length first
max_length = 1000   # 1727 for S645
input_dict = {}
wt_dict = {}

for pdb_name in pdbs:
    try:
        wild_type_file = pdb_name + '_wt_encoded.npy'
        encoding_file = pdb_name + '_encoded.csv'
        x_wild_type=np.load(wild_type_file)
        df_encoding=pd.read_csv(encoding_file, sep=';')
    except:
        break

    variants = df_encoding.iloc[:,0]
    X_values = df_encoding.iloc[:,2:].to_numpy()
    Ys = df_encoding.iloc[:,1]

    var_num = X_values.shape[0]
    seq_len = X_values.shape[1]
    
    # mutants = variants.str[1:-1]
    for var in range(var_num):
        # res_num = int(res_num) - 1
        encoding_vector = X_values[var]
        input_X = np.zeros((1, max_length))
        input_X[0:(var+1), :seq_len] = encoding_vector
        key = pdb_name + '_' + str(var - 1)
        input_dict[key] = {'pdb_name':pdb_name, 'length':seq_len, 'variants':variants[var], 'y':Ys[var], 'Xs':input_X}
    # save wt encodings
    input_wt = np.zeros((1, max_length))
    input_wt[0:, :seq_len] = x_wild_type
    wt_dict[pdb_name] = input_wt

with open('wt_encoded.pkl', 'wb') as wtf:
    pickle.dump(wt_dict, wtf)
with open('input500.pkl', 'wb') as f:
    pickle.dump(input_dict, f)
