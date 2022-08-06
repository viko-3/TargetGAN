import json

import pandas as pd
import os
import torch


def get_smiles_and_uid_smi(df):
    uniprot_id = df['uniprot_ID']
    smiles = df['smiles']
    storage_path = '/home/chenyy/Code/latent-gan/data/Agument_CPI/'
    with open(os.path.join(storage_path, 'smiles.smi'), 'w') as f:
        for s in smiles:
            f.write(s + '\n')
    with open(os.path.join(storage_path, 'uniprot_ID.smi'), 'w') as f:
        for id in uniprot_id:
            f.write(id + '\n')


def get_proteins_feature(all_uniprot_id):
    feature_matrix = torch.zeros((len(all_uniprot_id), 768))
    proteins_feature_path = '/home/chenyy/Code/latent-gan/data/Agument_CPI/proteins'
    for index, uniprot_id in enumerate(all_uniprot_id):
        uniprot_path = os.path.join(proteins_feature_path, '{}.npz'.format(uniprot_id))
        feature = torch.load(uniprot_path)
        feature_matrix[index] = feature
    print('ok')
    res = feature_matrix.tolist()
    del feature_matrix
    encoded_proteins_path = '/home/chenyy/Code/latent-gan/storage/Augment_CPI/encoded_proteins.latent'
    with open(encoded_proteins_path, 'w') as f:
        json.dump(res, f)



data_csv = '/home/chenyy/Code/latent-gan/data/Agument_CPI/aug_label_1_process_data.csv'
df = pd.read_csv(data_csv)

all_uniprot_id = df['uniprot_ID']
get_smiles_and_uid_smi(df)
del df
print(all_uniprot_id[0])
exit()
get_proteins_feature(all_uniprot_id)

