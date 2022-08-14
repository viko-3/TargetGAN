import torch
import torch.nn as nn
import numpy as np



class Discriminator(nn.Module):
    def __init__(self, data_shape):
        super(Discriminator, self).__init__()
        self.data_shape = data_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.data_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, mol):
        validity = self.model(mol)
        return validity

    def save(self, path):
        save_dict = {
            'model': self.model.state_dict(),
            'data_shape': self.data_shape,
        }
        torch.save(save_dict, path)
        return

    @staticmethod
    def load(path):
        save_dict = torch.load(path)
        D = Discriminator(save_dict['data_shape'])
        D.model.load_state_dict(save_dict["model"])

        return D


class C_Discriminator(nn.Module):
    def __init__(self, data_shape):
        super(C_Discriminator, self).__init__()
        self.data_shape = data_shape
        self.condition_dim = 768

        self.condition_MLE = nn.Linear(768, 512)
        self.model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, mol, condition):
        condition = condition.to(torch.float32)
        condition = self.condition_MLE(condition)
        feature = torch.cat((mol, condition), dim=1)
        validity = self.model(feature)
        return validity

    def save(self, path):
        save_dict = {
            'model': self.model.state_dict(),
            'data_shape': self.data_shape,
        }
        torch.save(save_dict, path)
        return

    @staticmethod
    def load(path):
        save_dict = torch.load(path)
        D = C_Discriminator(save_dict['data_shape'])
        D.model.load_state_dict(save_dict["model"])

        return D
