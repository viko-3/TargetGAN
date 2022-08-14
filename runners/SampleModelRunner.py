from models.Generator import Generator
from src.Sampler import Sampler
import torch
import json


class SampleModelRunner:

    def __init__(self, output_latent_file, input_model_path, sample_number):
        # init params
        self.input_model_path = input_model_path
        self.output_latent_file = output_latent_file
        self.sample_number = sample_number

        self.G = Generator.load(input_model_path)

        # Tensor
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.G.cuda()
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def run(self):
        # sampling mode
        torch.no_grad()
        self.G.eval()

        S = Sampler(generator=self.G)
        latent = S.sample(self.sample_number)
        latent = latent.detach().cpu().numpy().tolist()

        with open(self.output_latent_file, 'w') as json_file:
            json.dump(latent, json_file)
