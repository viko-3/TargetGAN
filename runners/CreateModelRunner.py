from models.Discriminator import Discriminator, C_Discriminator
from models.Generator import Generator, C_Generator
import os
import numpy as np
import json


class CreateModelRunner:
    def __init__(self, input_data_path, output_model_folder):
        self.input_data_path = input_data_path
        self.output_model_folder = output_model_folder
        # get data
        latent_vector_file = open(self.input_data_path, "r")
        latent_space_mols = np.array(json.load(latent_vector_file))
        shape = latent_space_mols.shape  # expecting tuple (set_size, dim_1, dim_2)
        assert len(shape) == 3
        self.data_shape = tuple([shape[1], shape[2]])

    def run(self):
        self.c_CreateGenerator()
        self.CreateDiscriminator()
        self.c_CreateDiscriminator()

    def CreateDiscriminator(self):
        # create Discriminator
        D = Discriminator(self.data_shape)

        # save Discriminator
        if not os.path.exists(self.output_model_folder):
            os.makedirs(self.output_model_folder)
        discriminator_path = os.path.join(self.output_model_folder, 'discriminator.txt')
        D.save(discriminator_path)

    def CreateGenerator(self):
        # create Generator
        G = Generator(self.data_shape, latent_dim=self.data_shape[1])

        # save generator
        if not os.path.exists(self.output_model_folder):
            os.makedirs(self.output_model_folder)
        generator_path = os.path.join(self.output_model_folder, 'generator.txt')
        G.save(generator_path)

    # 新加的{
    def c_CreateGenerator(self):
        # create Generator
        G = C_Generator(self.data_shape, latent_dim=self.data_shape[1])

        # save generator
        if not os.path.exists(self.output_model_folder):
            os.makedirs(self.output_model_folder)
        generator_path = os.path.join(self.output_model_folder, 'generator.txt')
        G.save(generator_path)

    def c_CreateDiscriminator(self):
        # create Discriminator
        D = C_Discriminator(self.data_shape)

        # save Discriminator
        if not os.path.exists(self.output_model_folder):
            os.makedirs(self.output_model_folder)
        discriminator_path = os.path.join(self.output_model_folder, 'c_discriminator.txt')
        D.save(discriminator_path)
