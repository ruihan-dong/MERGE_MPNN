# split file
import os
import pandas as pd
from Bio import PDB
from graphein.protein import graphs

S645_data = pd.read_excel('../PPI.xlsx', sheet_name='S645')
# S1131_data = pd.read_excel('../PPI.xlsx', sheet_name='S1131')
pdb_chain = S645_data.iloc[:,0:2].drop_duplicates()
pdb_chain['Partners(A_B)'] = pdb_chain['Partners(A_B)'].str.replace('_','')
pdb_chain['Partners(A_B)'] = pdb_chain['Partners(A_B)'].apply(lambda x: ' '.join(str(x)))

for index,row in pdb_chain.iterrows():
    pdb_name = row[0]
    pdb_data = S1131_data[S1131_data['protein'] == pdb_name]
    designed_chains = row[1].split(' ')

    df_mut = pdb_data['Mutation']
    df_ddg = pdb_data['ddG(kcal/mol)']
    df_chain = df_mut.str[0]
    df_mutants = df_mut.str[2:]  # e.g. F17A
    df_before_aa = df_mutants.str[0:1]  # F
    df_after_aa = df_mutants.str[-1:]  # A
    df_num = df_mutants.str[1:-1]   # 17

    df_pdb = graphs.read_pdb_to_dataframe(pdb_path = '../S645-pdb/'+pdb_name+'.pdb')
    df_pdb_chains = df_pdb.loc[df_pdb['chain_id'].isin(designed_chains)]
    df_ca = df_pdb_chains.loc[df_pdb_chains['atom_name'] == 'CA'].reset_index(drop=True)

    new_mutants = []
    new_ddg = []
    for i in df_mutants.index:
        try:
            chain = df_chain[i]
            num = df_num[i]
            ddg = df_ddg[i]

            # for 100a/100b
            num_insert = num[-1:]
            if num_insert == 'a' or num_insert == 'b' or num_insert == 'c' or num_insert == 'd' or num_insert == 'e' or num_insert == 'f' or num_insert == 'g':
                num = num[:-1]
            else:
                num_insert = ''

            new_num = df_ca[(df_ca['residue_number'] == int(num)) & (df_ca['chain_id'] == chain)].index[0]
            new_num = int(new_num) + 1

            new_mutants.append(df_before_aa[i] + str(new_num) + num_insert + df_after_aa[i])
            new_ddg.append(ddg)
        except:
            print('error: ', df_mutants[i])

    df_new_mutants = pd.DataFrame({'variant':new_mutants, 'y':new_ddg})
    df_new_mutants.to_csv(pdb_name + '.csv', sep=';', index=None)
    try:
        os.system('python ~/Downloads/MERGE-main/Examples/scripts/make_dataframe.py \
            -start_pos 1 -params ./params500/' + pdb_name +'_plmc.params \
            -fitness y -csv ' + pdb_name + '.csv')
    except:
        print('error: ', pdb_name)
