from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import re

def filter_invalid(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        return smi
    else:
        return None

def smiles2mol(smiles):
    m=Chem.MolFromSmiles(smiles)
    #print(Chem.MolToMolBlock(m)) 
    AllChem.Compute2DCoords(m)
    return m

path = '/home/chenyy/Code/latent-gan/storage/CPI/proteins/P50406.csv'
df = pd.read_csv(path)
df['SMILES'] = df['SMILES'].apply(filter_invalid)
df = df.dropna()
df.to_csv(path,index = False)
exit()
df['MOL'] = df['SMILES'].apply(smiles2mol)
for i, mol in enumerate(df['MOL']):
    Chem.MolToMolFile(mol,'/home/chenyy/Code/latent-gan/utils/P09874_gen/{}.sdf'.format(i))