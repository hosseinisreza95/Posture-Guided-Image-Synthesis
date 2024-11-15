import numpy as np
import cv2
import os
import pickle
import sys
import math

import matplotlib.pyplot as plt

from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenVanillaNN import *


class Discriminator(nn.Module):
    def __init__(self, ngpu=0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        # Input: (batch_size, 3, 64, 64)
        self.model = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # 1x1
        )

        # Initialize weights
        self.apply(init_weights)

    def forward(self, x):
        return self.model(x).view(-1)


class GenGAN():
    def __init__(self, videoSke, loadFromFile=False):
        self.netG = GenNNSkeToImage()
        self.netD = Discriminator()
        self.real_label = 1.
        self.fake_label = 0.
        self.filename = 'data/Dance/DanceGenGAN.pth'

        tgt_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform)
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=64,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        if loadFromFile and os.path.isfile(self.filename):
            print(f"GenGAN: Loading from {self.filename}")
            checkpoint = torch.load(self.filename)
            self.netG.load_state_dict(checkpoint['generator_state_dict'])
            self.netD.load_state_dict(checkpoint['discriminator_state_dict'])

    def add_noise(self, images, noise_factor=0.05):
        noise = torch.randn_like(images) * noise_factor
        noisy_images = images + noise
        return torch.clamp(noisy_images, -1, 1)

    def train(self, n_epochs=20):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

        # Move models to GPU
        self.netG = self.netG.to(device)
        self.netD = self.netD.to(device)

        # Initialize optimizers with different learning rates
        optimizerG = torch.optim.Adam(self.netG.parameters(), lr=0.004,
                                      betas=(0.5, 0.999))  # Higher learning rate for G
        optimizerD = torch.optim.Adam(self.netD.parameters(), lr=0.001,
                                      betas=(0.5, 0.999))  # Lower learning rate for D

        # Initialize schedulers
        schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerG, mode='min', factor=0.5, patience=5,
                                                                verbose=True)
        schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerD, mode='min', factor=0.5, patience=5,
                                                                verbose=True)

        # Initialize gradient scalers
        scalerG = GradScaler()
        scalerD = GradScaler()

        # Loss functions
        criterion_GAN = nn.BCEWithLogitsLoss()
        criterion_pixel = nn.MSELoss()

        # Training hyperparameters
        NOISE_FACTOR = 0.005
        D_TRAINING_FREQ = 10  # Train D every n iterations
        LABEL_SMOOTHING = 0.8

        print(f"Starting training on {device} for {n_epochs} epochs...")
        for epoch in range(n_epochs):
            total_d_loss = 0
            total_g_loss = 0

            for i, (skeletons, real_images) in enumerate(self.dataloader):
                batch_size = real_images.size(0)

                # Move data to device
                real_images = real_images.to(device, non_blocking=True)
                skeletons = skeletons.to(device, non_blocking=True)

                # Add noise to real images
                real_images_noisy = self.add_noise(real_images, NOISE_FACTOR)

                # Labels with smoothing
                real_target = torch.ones(batch_size, device=device) * (1 - LABEL_SMOOTHING)
                fake_target = torch.zeros(batch_size, device=device) + LABEL_SMOOTHING

                ############################
                # Train Discriminator (less frequently)
                ############################
                if i % D_TRAINING_FREQ == 0:
                    self.netD.zero_grad(set_to_none=True)

                    with autocast():
                        # Real images
                        output_real = self.netD(real_images_noisy)
                        d_loss_real = criterion_GAN(output_real, real_target)

                        # Fake images
                        fake_images = self.netG(skeletons)
                        fake_images_noisy = self.add_noise(fake_images.detach(), NOISE_FACTOR)
                        output_fake = self.netD(fake_images_noisy)
                        d_loss_fake = criterion_GAN(output_fake, fake_target)

                        d_loss = (d_loss_real + d_loss_fake) * 0.5

                    scalerD.scale(d_loss).backward()
                    scalerD.step(optimizerD)
                    scalerD.update()

                ############################
                # Train Generator (more frequently)
                ############################
                self.netG.zero_grad(set_to_none=True)

                with autocast():
                    # Generate new fake images
                    fake_images = self.netG(skeletons)

                    # GAN loss with noise
                    fake_images_noisy = self.add_noise(fake_images, NOISE_FACTOR)
                    output_fake = self.netD(fake_images_noisy)
                    g_loss_GAN = criterion_GAN(output_fake, real_target)

                    # Pixel-wise loss
                    g_loss_pixel = criterion_pixel(fake_images, real_images)

                    # Combined loss (with higher weight on pixel loss)
                    g_loss = g_loss_GAN + 150 * g_loss_pixel  # Increased pixel loss weight

                scalerG.scale(g_loss).backward()
                scalerG.step(optimizerG)
                scalerG.update()

                # Print progress
                if i % 10 == 0:
                    print(f'[{epoch + 1}/{n_epochs}][{i}/{len(self.dataloader)}] '
                          f'Loss_D: {d_loss.item():.4f} Loss_G: {g_loss.item():.4f}')

                # Clear cache periodically
                if i % 50 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()

            # Update schedulers
            schedulerG.step(g_loss)
            schedulerD.step(d_loss)

            # Save checkpoint periodically
            if (epoch + 1) % 5 == 0:
                print(f"Saving model at epoch {epoch + 1}...")
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': self.netG.state_dict(),
                    'discriminator_state_dict': self.netD.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    'schedulerG_state_dict': schedulerG.state_dict(),
                    'schedulerD_state_dict': schedulerD.state_dict(),
                    'g_loss': g_loss.item(),
                    'd_loss': d_loss.item(),
                }, self.filename)

    def generate(self, ske):
        device = next(self.netG.parameters()).device

        ske_t = self.dataset.preprocessSkeleton(ske)
        ske_t_batch = ske_t.unsqueeze(0)
        ske_t_batch = ske_t_batch.to(device)

        self.netG.eval()
        with torch.no_grad():
            with autocast():
                generated_image = self.netG(ske_t_batch)
            generated_image = generated_image.cpu()
            result = self.dataset.tensor2image(generated_image[0])

        return result


if __name__ == '__main__':
    force = False
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"

    print("GenGAN: Current Working Directory =", os.getcwd())
    print("GenGAN: Filename =", filename)

    targetVideoSke = VideoSkeleton(filename)

    if True:
        gen = GenGAN(targetVideoSke, loadFromFile=False)
        gen.train(1000)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)

    # Test generation
    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        image = cv2.resize(image, (256, 256))
        cv2.imshow('Generated Image', image)
        key = cv2.waitKey(-1)