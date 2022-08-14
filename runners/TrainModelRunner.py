import pickle

from datasets.TargetMolsDataset import TargetMolsDataset
from models.Discriminator import Discriminator, C_Discriminator
from models.Generator import Generator, C_Generator
from datasets.LatentMolsDataset import LatentMolsDataset
from src.Sampler import Sampler, C_Sampler
from decode import decode
import os
import torch
import torch.autograd as autograd
import numpy as np
import json
import time
import sys
from tqdm import tqdm
import pandas as pd
import logging


class TrainModelRunner:
    # Loss weight for gradient penalty

    def __init__(self, input_data_path, output_model_folder, decode_mols_save_path='', n_epochs=2000, starting_epoch=1,
                 batch_size=4096, lr=0.0001, b1=0.5, b2=0.999, lambda_gp=10, n_critic=5,
                 save_interval=500, sample_after_training=30000, message="", decoder=None):
        self.message = message

        # init params
        self.input_data_path = input_data_path
        self.output_model_folder = output_model_folder
        self.n_epochs = n_epochs
        self.starting_epoch = starting_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        self.save_interval = save_interval
        self.sample_after_training = sample_after_training
        self.decode_mols_save_path = decode_mols_save_path
        self.decoder = decoder

        # initialize dataloader
        json_smiles = open(self.input_data_path, "r")
        latent_space_mols = np.array(json.load(json_smiles))
        latent_space_mols = latent_space_mols.reshape(latent_space_mols.shape[0], 512)
        """
        # 之前的
        self.dataloader = torch.utils.data.DataLoader(LatentMolsDataset(latent_space_mols), shuffle=True,
                                                      batch_size=self.batch_size, drop_last=True)
        """
        # 新加的{
        json_proteins = './storage/split_CPI_PARP1/encoded_proteins.latent'
        json_proteins = open(json_proteins, "r")
        proteins_feature = np.array(json.load(json_proteins))
        """# 扩充两倍
        proteins_feature = np.concatenate((proteins_feature,proteins_feature),axis=0)
        #"""
        self.dataloader = TargetMolsDataset(latent_space_mols=latent_space_mols, proteins_feature=proteins_feature)


        # load discriminator
        discriminator_name = 'discriminator.txt' if self.starting_epoch == 1 else str(
            self.starting_epoch - 1) + '_discriminator.txt'
        discriminator_path = os.path.join(output_model_folder, discriminator_name)
        self.D = Discriminator.load(discriminator_path)
        # 新加的{
        discriminator_name = 'c_discriminator.txt' if self.starting_epoch == 1 else str(
            self.starting_epoch - 1) + 'c_discriminator.txt'
        discriminator_path = os.path.join(output_model_folder, discriminator_name)
        self.c_D = C_Discriminator.load(discriminator_path)
        # }

        # load generator
        generator_name = 'generator.txt' if self.starting_epoch == 1 else str(
            self.starting_epoch - 1) + '_generator.txt'
        generator_path = os.path.join(output_model_folder, generator_name)
        self.G = C_Generator.load(generator_path)

        # initialize sampler
        self.Sampler = C_Sampler(self.G)

        # initialize optimizer
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_c_D = torch.optim.Adam(self.c_D.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        # Tensor
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.G.cuda()
            self.D.cuda()
            self.c_D.cuda()
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def run(self):
        # self.test()
        # exit()
        print("Training of GAN started.")
        print("Message: %s" % self.message)
        sys.stdout.flush()

        batches_done = 0
        disc_loss_log = []
        g_loss_log = []

        for epoch in range(self.starting_epoch, self.n_epochs + self.starting_epoch):
            disc_loss_per_batch = []
            g_loss_log_per_batch = []
            # pre:
            # for i, real_mols in enumerate(tqdm(self.dataloader)):
            for i, data in enumerate(tqdm(self.dataloader.loader)):
                real_mols, proteins = data['mols'], data['proteins']
                proteins = proteins.cuda()
                # Configure input
                real_mols = real_mols.type(self.Tensor)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Generate a batch of mols from noise
                fake_mols = self.Sampler.sample(real_mols.shape[0], proteins)

                # Real mols
                real_validity = (self.D(real_mols) + self.c_D(real_mols, proteins)) / 2
                # Fake mols
                fake_validity = (self.D(fake_mols) + self.c_D(fake_mols, proteins)) / 2

                # Gradient penalty
                gradient_penalty = self.compute_gradient_penalty(real_mols.data, fake_mols.data, proteins)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty
                disc_loss_per_batch.append(d_loss.item())

                d_loss.backward()
                self.optimizer_D.step()
                self.optimizer_G.zero_grad()
                self.optimizer_c_D.zero_grad()

                # Train the generator every n_critic steps
                if i % self.n_critic == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of mols
                    fake_mols = self.Sampler.sample(real_mols.shape[0], proteins)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = (self.D(fake_mols) + self.c_D(fake_mols, proteins)) / 2
                    g_loss = -torch.mean(fake_validity)
                    g_loss_log_per_batch.append(g_loss.item())

                    g_loss.backward()
                    self.optimizer_G.step()

                    batches_done += self.n_critic

                # If last batch in the set
                if i == len(self.dataloader.loader) - 1:
                    if epoch % self.save_interval == 0:
                        generator_save_path = os.path.join(self.output_model_folder,
                                                           str(epoch) + '_generator.txt')
                        discriminator_save_path = os.path.join(self.output_model_folder,
                                                               str(epoch) + '_discriminator.txt')
                        c_discriminator_save_path = os.path.join(self.output_model_folder,
                                                                 str(epoch) + 'c_discriminator.txt')
                        self.G.save(generator_save_path)
                        self.D.save(discriminator_save_path)
                        self.c_D.save(c_discriminator_save_path)

                    disc_loss_log.append([time.time(), epoch, np.mean(disc_loss_per_batch)])
                    g_loss_log.append([time.time(), epoch, np.mean(g_loss_log_per_batch)])

                    # Print and log
                    print(
                        "[Epoch %d/%d]  [Disc loss: %f] [Gen loss: %f] "
                        % (epoch, self.n_epochs + self.starting_epoch, disc_loss_log[-1][2], g_loss_log[-1][2])
                    )
                    sys.stdout.flush()

        # log the losses
        with open(os.path.join(self.output_model_folder, 'disc_loss.json'), 'w') as json_file:
            json.dump(disc_loss_log, json_file)
        with open(os.path.join(self.output_model_folder, 'gen_loss.json'), 'w') as json_file:
            json.dump(g_loss_log, json_file)

        # Sampling after training
        if self.sample_after_training > 0:
            print("Training finished. Generating sample of latent vectors")

            # sampling mode
            torch.no_grad()
            self.G.eval()
            for i, data in enumerate(tqdm(self.dataloader.loader)):
                real_mols, proteins = data['mols'], data['proteins']
                proteins = proteins.cuda()
                latent = self.Sampler.sample(
                    self.sample_after_training if self.sample_after_training == len(proteins) else len(proteins),
                    proteins)
                latent = latent.detach().cpu().numpy().tolist()

                sampled_mols_save_path = os.path.join(self.output_model_folder, 'sampled.json')
                with open(sampled_mols_save_path, 'w') as json_file:
                    json.dump(latent, json_file)

                # decoding sampled mols
                print("Sampling done. Decoding latent vectors into SMILES")
                decode(sampled_mols_save_path, self.decode_mols_save_path, model=self.decoder)
                break

    def compute_gradient_penalty(self, real_samples, fake_samples, condition):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = self.Tensor(np.random.random((real_samples.size(0), 1)))

        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = (self.D(interpolates) + self.c_D(interpolates, condition)) / 2
        fake = self.Tensor(real_samples.shape[0], 1).fill_(1.0)

        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def test(self):
        generator_path = '/home/s2136015/Code/latent-gan/storage/split_CPI_PARP1/2000_generator.txt'
        self.G = C_Generator.load(generator_path)
        self.G.cuda()
        self.Sampler = C_Sampler(self.G)
        self.G.eval()
        self.condition_sample()
        # self.sample_all()

    def condition_sample(self):
        """
        # sampled for MOSES  number:30000
        moses_latent = torch.zeros((30000, 512)).cuda()
        for i, data in enumerate(tqdm(self.dataloader.loader)):
            real_mols, proteins = data['mols'], data['proteins']
            proteins = proteins.cuda()
            latent = self.Sampler.sample(
                self.sample_after_training if self.sample_after_training == len(proteins) else len(proteins),
                proteins)
            if (i + 1) * len(proteins) > 30000:
                num = 30000 - i * len(proteins)
                moses_latent[i * len(proteins):] = latent[:num]
                break
            moses_latent[i * len(proteins):(i + 1) * len(proteins)] = latent

        latent = moses_latent.detach().cpu().numpy().tolist()

        sampled_mols_save_path = os.path.join(self.output_model_folder, 'moses_sampled.json')
        with open(sampled_mols_save_path, 'w') as json_file:
            json.dump(latent, json_file)

        # decoding sampled mols
        print("MOSES Sampling done. Decoding latent vectors into SMILES")
        decode(sampled_mols_save_path, self.decode_mols_save_path, model=self.decoder)
        exit()
        """
        
        # sampled for DTA
        uniprot_id__csv = '/home/s2136015/Code/DataSet/New_CPI/test_proteins.csv'
        df = pd.read_csv(uniprot_id__csv)
        all_uniprot_id = df['uniprot_ID']
        proteins_path = './data/CPI/proteins'
        for index, uniprot_id in enumerate(all_uniprot_id):
            uniprot_id = 'P09874'
            uniprot_path = os.path.join(proteins_path, '{}.npz'.format(uniprot_id))
            print(uniprot_id)
            feature = torch.load(uniprot_path)
            proteins = feature.repeat(100000, 1)
            proteins = proteins.cuda()
            gen_mols = self.Sampler.sample(proteins.shape[0], proteins)
            gen_mols = gen_mols.detach().cpu().numpy().tolist()
            sampled_mols_save_path = os.path.join(self.output_model_folder, 'proteins', '{}.json'.format(uniprot_id))
            with open(sampled_mols_save_path, 'w') as json_file:
                json.dump(gen_mols, json_file)

            # decoding sampled mols
            decode_mols_save_path = os.path.join(self.output_model_folder, 'proteins', '{}.csv'.format(uniprot_id))
            decode(sampled_mols_save_path, decode_mols_save_path, model=self.decoder)
            break

    def sample_all(self):
        for i, data in enumerate(tqdm(self.dataloader.loader)):
            real_mols, proteins = data['mols'], data['proteins']
            proteins = proteins.cuda()
            latent = self.Sampler.sample(
                    self.sample_after_training if self.sample_after_training == len(proteins) else len(proteins),
                    proteins)
            latent = latent.detach().cpu().numpy().tolist()

            sampled_mols_save_path = os.path.join(self.output_model_folder, 'sampled.json')
            with open(sampled_mols_save_path, 'w') as json_file:
                json.dump(latent, json_file)

                # decoding sampled mols
            print("Sampling done. Decoding latent vectors into SMILES")
            decode(sampled_mols_save_path, self.decode_mols_save_path, model=self.decoder)
