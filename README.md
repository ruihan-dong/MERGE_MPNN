# MERGE_MPNN
Structure-bridged Protein-Protein Binding Affinity Changes upon Mutations via Designed Sequences

This is a simple trial during my third rotation. Discussions are welcomed via emailing **dongruihan_at_stu.pku.edu.cn**.

## Datasets
PPI.xlsx
S645 single mutation with ddG from AB-Bind. PDB stuctures can be found [here](https://github.com/sarahsirin/AB-Bind-Database). 

## Dependencies
- graphein ≤ 1.5.0
- openpyxl
- xlrd == 1.2.0
- python ≥ 3.7
- sklearn
- pytorch (for ProteinMPNN)

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
**WARNING**: This step may need long time to run. You can modify the parameters of plmc, i.e. '-m' for smaller maximal iterations. 

For sequences whose length is longer than 1000, more storage is needed. 

### Split dataframe
Now the direct couplings of each complex is obtained. In this step, two scripts help to transform .pkl into the model input, and combine the input files of each structure.
```bash
python split_dataframe.py
python merge_encodings.py
```

### Run model
This is a modified version of final predicting model. I change the ridge regressor of original MERGE framework to a simple MLP. And 90/10 train/test split ratio is used here. More regression metrics are added. 
```bash
python run_model.py
```

## Results
Pearson on 10-fold validation is 0.47.

Comparision with other ddG prediction models:
![image](https://github.com/ruihan-dong/MERGE_MPNN/blob/main/comparison.png)
