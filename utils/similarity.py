from rdkit import Chem, DataStructs
import pandas as pd
from rdkit.Chem import AllChem, Draw

def TAanimoto_Similarity(smiles1, smiles2):
    # print(smiles1)
    # print(smiles2)
    mols = [Chem.MolFromSmiles(smiles1), Chem.MolFromSmiles(smiles2)]
    """bv = [Chem.RDKFingerprint(x) for x in mols]
    similarity = DataStructs.TanimotoSimilarity(bv[0], bv[1])"""
    morgan_fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2,1024) for mol in mols]
    similarity = DataStructs.BulkTanimotoSimilarity(morgan_fps[0], morgan_fps[1:])
    return similarity[0]

def single_find(smiles_list):
    smiles = 'NC(=O)c1ccc2c(c1)[nH]c(=O)c1ccsc12'
    res_path = './single_smilarity2.txt'
    f = open(res_path,'w')
    for smiles2 in smiles_list:
        smilarity = TAanimoto_Similarity(smiles,smiles2)
        if smilarity>0.4:
            f.write(smiles2+' '+str(smilarity))
            f.write('\n')
    f.close()

def loop_find(smiles_list1,smiles_list2):
    res_path = './P14416_smilarity2.txt'
    f = open(res_path,'w') 
    for smiles1 in smiles_list1:
        # print(smiles1)
        max_smilarity = 0
        max_smiles = ''
        for smiles2 in smiles_list2:
            smilarity = TAanimoto_Similarity(smiles1,smiles2)
            max_smiles = smiles2 if smilarity>max_smilarity else max_smiles
            max_smilarity = smilarity if smilarity>max_smilarity else max_smilarity
        print(max_smilarity)
        f.write(smiles1+' '+ max_smiles+' '+str(max_smilarity))
        f.write('\n')
    f.close()
    

if __name__ == '__main__':
    """path = '/home/chenyy/Code/data/103_data/Q99705.csv'
    df = pd.read_csv(path)
    all_smiles = df['SMILES'][1:]
    smiles1 = 'NC(NC(Nc1c(CS(=O)Nc2ccccc2)cccc1)CCO)CNC(=O)Nc1ccc(C#N)cc1'
    # smiles2 = 'CC(C)C(O)C(=O)NC1CCC(CCN2CCN(c3cccc4c3OCO4)CC2)CC1'
    all_similarity = []
    for s in all_smiles:
        res = TAanimoto_Similarity(smiles1, s)
        all_similarity.append(res)
    print(sorted(all_similarity,reverse=False))"""
    path1 = '/home/chenyy/Code/latent-gan/storage/CPI/proteins/P14416_real.csv'
    path2 = '/home/chenyy/Code/latent-gan/storage/CPI/proteins/P14416.csv'
    df1,df2 = pd.read_csv(path1),pd.read_csv(path2)
    all_smiles1,all_smiles2 = df1['SMILES'],df2['SMILES']
    res=loop_find(all_smiles1,all_smiles2)
    # single_find(all_smiles2)
    
