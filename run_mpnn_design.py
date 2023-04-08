import os
import pandas as pd
S645_data = pd.read_excel('PPI.xlsx', sheet_name='S645')
# S1131_data = pd.read_excel('PPI.xlsx', sheet_name='S1131')
pdbs = S645_data.iloc[:,0:2].drop_duplicates()
pdbs['Partners(A_B)'] = pdbs['Partners(A_B)'].str.replace('_','')
pdbs['Partners(A_B)'] = pdbs['Partners(A_B)'].apply(lambda x: ' '.join(str(x)))

output_dir = './S645_outputs/seq500/'
for index,row in pdbs.iterrows():
    path_to_PDB = './S645-pdb/' + row[0] + '.pdb'
    chains_to_design = '"' + row[1] + '"'
    os.system('python ./ProteinMPNN/protein_mpnn_run.py \
        --pdb_path ' + path_to_PDB + 
        ' --pdb_path_chains ' + chains_to_design +
        ' --out_folder ' + output_dir + \
        ' --num_seq_per_target 500 \
        --sampling_temp "0.1" \
        --seed 37 \
        --batch_size 1')
    print(row[0] + ' finished')
