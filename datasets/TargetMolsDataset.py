import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np


class GANDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class TargetMolsDataset(Dataset):
    def __init__(self, latent_space_mols, proteins_feature, batch_size=1024, shuffle=True):
        self.latent_space_mols = latent_space_mols
        self.proteins_feature = proteins_feature
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.loader = DataLoader(
            dataset=GANDataset(self.prepare_data(self.latent_space_mols, self.proteins_feature)),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=False)

    def prepare_data(self, latent_space_mols, proteins_feature):
        data = [{'mols': smiles, 'proteins': proteins} for smiles, proteins in zip(latent_space_mols, proteins_feature)]
        return data
