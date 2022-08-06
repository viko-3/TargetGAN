from json.tool import main
from webbrowser import get
from tape import ProteinBertModel, TAPETokenizer
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
import torch.nn.functional as F

model = ProteinBertModel.from_pretrained('bert-base')
tokenizer = TAPETokenizer(vocab='iupac')



def proteins_encode(sequence):
    token_ids = torch.tensor([tokenizer.encode(sequence)])
    output = model(token_ids)
    sequence_output = output[0]
    sequence_output = torch.mean(sequence_output, dim=1)
    #return torch.squeeze(sequence_output)
    return sequence_output


def get_feature(data):
    for d in data.iterrows():
        d = d[1]
        uniprot_ID, sequence = d['uniprot_ID'], d['proteins']
        print(uniprot_ID)
        feature = proteins_encode(sequence)
        # print(feature.shape)
        res =torch.squeeze(feature)
        print(res.shape)
        # exit()
        torch.save(res, '/home/chenyy/Code/latent-gan/data/CPI/proteins/{}.npz'.format(uniprot_ID))


data_path = '/home/chenyy/Code/DataSet/New_CPI/test_proteins.csv'
df = pd.read_csv(data_path)
get_feature(df)
