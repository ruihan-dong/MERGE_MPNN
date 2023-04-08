import os
import pandas as pd
S645_data = pd.read_excel('../../PPI.xlsx', sheet_name='S645')
pdb = S645_data['#PDB'].drop_duplicates().to_list()

for pdb_name in pdb:
    # get sequence
    file_name = '../seq500/seqs/' + pdb_name + '.fa'
    with open(file_name, 'r') as f:
        fa = f.read()
        fa = fa.replace('/', '')  # remove "/" for multichain
        fa = fa.replace('X', '')  # remove "X"
        seq = fa.splitlines()[1]
        print(seq)
    with open(file_name, 'w') as fw:
        fw.write(fa)

    length = len(seq)
    le = 0.2 * (length - 1)

    os.system('~/Downloads/plmc-master/bin/plmc -o ' + pdb_name + '_plmc.params' \
        ' -le ' + str(le) + \
        ' -lh 0.01 -m 100 -g -f ' + pdb_name + ' ' + file_name)
