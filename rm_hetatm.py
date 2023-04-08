# https://stackoverflow.com/questions/25718201/remove-heteroatoms-from-pdb

from Bio.PDB import PDBParser, PDBIO, Select

class NonHetSelect(Select):
    def accept_residue(self, residue):
        return 1 if residue.id[0] == " " else 0

with open('list','r') as list_file:
    pdb_list = list_file.read().splitlines()

for pdb_file in pdb_list:
    pdb_name = pdb_file[-7:]
    pdb = PDBParser().get_structure(pdb_name, pdb_file)
    io = PDBIO()
    io.set_structure(pdb)
    io.save(pdb_file, NonHetSelect())
