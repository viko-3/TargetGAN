import torch.nn as nn
import numpy as np
import torch


class Generator(nn.Module):
    def __init__(self, data_shape, latent_dim=None):
        super(Generator, self).__init__()
        self.data_shape = data_shape

        # latent dim of the generator is one of the hyperparams.
        # by default it is set to the prod of data_shapes
        self.latent_dim = int(np.prod(self.data_shape)) if latent_dim is None else latent_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.data_shape))),
            # nn.Tanh() # expecting latent vectors to be not normalized
        )

    def forward(self, z):
        out = self.model(z)
        return out

    def save(self, path):
        save_dict = {
            'latent_dim': self.latent_dim,
            'model': self.model.state_dict(),
            'data_shape': self.data_shape,
        }
        torch.save(save_dict, path)

        return

    @staticmethod
    def load(path):
        save_dict = torch.load(path)
        G = Generator(save_dict['data_shape'], latent_dim=save_dict['latent_dim'])
        G.model.load_state_dict(save_dict["model"])

        return G


class C_Generator(nn.Module):
    def __init__(self, data_shape, latent_dim=None):
        super(C_Generator, self).__init__()
        self.data_shape = data_shape

        # latent dim of the generator is one of the hyperparams.
        # by default it is set to the prod of data_shapes
        self.condition_dim = 768
        self.latent_dim = 512

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.condition_MLE = nn.Sequential(*block(self.condition_dim, 512))

        self.model = nn.Sequential(
            *block(self.latent_dim, 512, normalize=False),
            *block(512, 256),
            nn.Linear(256, int(np.prod(self.data_shape)))
            # nn.Tanh() # expecting latent vectors to be not normalized
        )

    def forward(self, z, condition):
        condition = condition.to(torch.float32)
        condition = self.condition_MLE(condition)
        feature = z.add(condition)
        # feature = torch.cat((z, condition), dim=1)
        out = self.model(feature)
        return out

    def save(self, path):
        save_dict = {
            'latent_dim': self.latent_dim,
            'model': self.model.state_dict(),
            'data_shape': self.data_shape,
        }
        torch.save(save_dict, path)

        return

    @staticmethod
    def load(path):
        save_dict = torch.load(path)
        G = C_Generator(save_dict['data_shape'], latent_dim=save_dict['latent_dim'])
        G.model.load_state_dict(save_dict["model"])

        return G
