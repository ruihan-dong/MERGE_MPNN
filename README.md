# MERGE_MPNN
Structure-bridged Protein-Protein Binding Affinity Changes upon Mutations via Designed Sequences

This is a simple trial during my third rotation. Discussions are welcomed via emailing dongruihan_at_stu.pku.edu.cn.

## Datasets
PPI.xlsx
S645 single mutation with ddG from AB-Bind. PDB stuctures can be found here. 

## Dependencies
- graphein ≤ 1.5.0
- openpyxl
- xlrd == 1.2.0
- python ≥ 3.7
- sklearn
- torch

## Procedure
### PDB preprocessing
```bash
ls *.pdb > list
python rm_hetatm.py
rm list
```
PDB files after preprocessing should be copied into S645-pdb folder.

### Run ProteinMPNN
Make sure you have downloaded the ProteinMPNN codes and it can run smoothly.
Please check the path of ProteinMPNN, S645 data, and output directory before starting sequence design.
```bash
python run_mpnn_design.py
```

### Run plmc
The next steps are referred to the MERGE model. 
To extract the direct couplings from designed sequences, plmc is used here. 
```bash
python run_plmc.py
```
WARNING: This step may need long time to run. You can modify the parameters of plmc, i.e. '-m' for smaller maximal iterations. 
For sequences whose length is longer than 1000, more storage is needed. 

### Split dataframe
Now the direct couplings of each complex is obtained.

