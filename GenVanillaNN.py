import numpy as np
import cv2
import os
import pickle
import sys
import math

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.io import read_image

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms

#from tensorboardX import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton

torch.set_default_dtype(torch.float32)


class SkeToImageTransform:
    def __init__(self, image_size):
        self.imsize = image_size

    def __call__(self, ske):
        #image = Image.new('RGB', (self.imsize, self.imsize), (255, 255, 255))
        image = white_image = np.ones((self.imsize, self.imsize, 3), dtype=np.uint8) * 255
        ske.draw(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('Image', image)
        # key = cv2.waitKey(-1)
        return image



class VideoSkeletonDataset(Dataset):
    def __init__(self, videoSke, ske_reduced, source_transform=None, target_transform=None):
        """ videoSkeleton dataset:
                videoske(VideoSkeleton): video skeleton that associate a video and a skeleton for each frame
                ske_reduced(bool): use reduced skeleton (13 joints x 2 dim=26) or not (33 joints x 3 dim = 99)
        """
        self.videoSke = videoSke
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced
        print("VideoSkeletonDataset: ",
              "ske_reduced=", ske_reduced, "=(", Skeleton.reduced_dim, " or ",Skeleton.full_dim,")" )


    def __len__(self):
        return self.videoSke.skeCount()


    def __getitem__(self, idx):
        # prepreocess skeleton (input)
        reduced = True
        ske = self.videoSke.ske[idx]
        ske = self.preprocessSkeleton(ske)
        # prepreocess image (output)
        image = Image.open(self.videoSke.imagePath(idx))
        if self.target_transform:
            image = self.target_transform(image)
        return ske, image


    def preprocessSkeleton(self, ske):
        if self.source_transform:
            ske = self.source_transform(ske)
        else:
            ske = torch.from_numpy( ske.__array__(reduced=self.ske_reduced).flatten() )
            ske = ske.to(torch.float32)
            ske = ske.reshape( ske.shape[0],1,1)
        return ske


    def tensor2image(self, normalized_image):
        numpy_image = normalized_image.detach().numpy()
        # Réorganiser les dimensions (C, H, W) en (H, W, C)
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        # passage a des images cv2 pour affichage
        numpy_image = cv2.cvtColor(np.array(numpy_image), cv2.COLOR_RGB2BGR)
        denormalized_image = numpy_image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
        denormalized_output = denormalized_image * 1
        return denormalized_output


# Weight initialization function
def init_weights(m):
    """Initialize network weights using the DCGAN strategy"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Conv layers: mean=0.0, std=0.02
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # BatchNorm layers: mean=1.0, std=0.02
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GenNNSkeToImage(nn.Module):
    def __init__(self):
        super(GenNNSkeToImage, self).__init__()
        self.input_dim = Skeleton.reduced_dim  # 26 dimensions (13 joints × 2D)

        # Improved generator architecture
        self.model = nn.Sequential(
            # Input layer: 26 -> 256
            nn.Linear(26, 256),
            nn.BatchNorm1d(256),  # Add BatchNorm
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # Add dropout

            # Hidden layer: 256 -> 512
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),  # Add BatchNorm
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # Add dropout

            # Hidden layer: 512 -> 1024
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),  # Add BatchNorm
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # Add dropout

            # Output layer: 1024 -> 3*64*64
            nn.Linear(1024, 3 * 64 * 64),
            nn.Tanh()  # Output range [-1, 1]
        )

        # Initialize weights
        self.apply(init_weights)
        print(self.model)

    def forward(self, z):
        """
        Forward pass of generator
        Args:
            z: Input tensor of shape (batch_size, 26, 1, 1)
        Returns:
            Generated image of shape (batch_size, 3, 64, 64)
        """
        # Reshape input: (batch_size, 26, 1, 1) -> (batch_size, 26)
        z = z.view(z.size(0), -1)

        # Generate image: (batch_size, 26) -> (batch_size, 3*64*64)
        img = self.model(z)

        # Reshape to image format: (batch_size, 3*64*64) -> (batch_size, 3, 64, 64)
        img = img.view(img.size(0), 3, 64, 64)

        return img

class GenNNSkeImToImage(nn.Module):
    """Class that generates images from skeleton images.
    Input: Skeleton rendered as image (3, 64, 64)
    Output: RGB image (3, 64, 64)
    """

    def __init__(self):
        super(GenNNSkeImToImage, self).__init__()
        self.input_dim = Skeleton.reduced_dim

        # Using ConvTranspose2d approach for better image quality
        self.model = nn.Sequential(
            # Initial convolution to process input image
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 64x32x32
            nn.LeakyReLU(0.2, inplace=True),

            # Downsample
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128x16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256x8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Bottleneck
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 512x4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Upsample
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 3x64x64
            nn.Tanh()  # Output range [-1, 1]
        )

        print(self.model)

    def forward(self, x):
        # Input shape: (batch_size, 3, 64, 64)
        # Output shape: (batch_size, 3, 64, 64)
        return self.model(x)





class GenVanillaNN():
    """ class that Generate a new image from a new skeleton posture
        Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        image_size = 64
        if optSkeOrImage==1:
            self.netG = GenNNSkeToImage()
            src_transform = None
            self.filename = 'data/Dance/DanceGenVanillaFromSke.pth'
        else:
            self.netG = GenNNSkeImToImage()
            src_transform = transforms.Compose([ SkeToImageTransform(image_size),
                                                 transforms.ToTensor(),
                                                 #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 ])
            self.filename = 'data/Dance/DanceGenVanillaFromSkeim.pth'



        tgt_transform = transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            # [transforms.Resize((64, 64)),
                            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=src_transform)
        print("datasetfffffffffffff",  self.dataset)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=64, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenVanillaNN: Load=", self.filename)
            print("GenVanillaNN: Current Working Directory: ", os.getcwd())
            self.netG = torch.load(self.filename)

    def train(self, n_epochs=20):
        """Train the generator network"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.netG = self.netG.to(device)

        # Initialize optimizer
        optimizer = torch.optim.Adam(self.netG.parameters(), lr=0.002, betas=(0.5, 0.999))
        criterion = nn.MSELoss()  # Mean squared error loss

        print(f"Starting training for {n_epochs} epochs...")
        for epoch in range(n_epochs):
            total_loss = 0
            for i, (skeletons, real_images) in enumerate(self.dataloader):
                # Move data to device
                skeletons = skeletons.to(device)
                real_images = real_images.to(device)

                # Generate images
                generated_images = self.netG(skeletons)

                # Calculate loss
                loss = criterion(generated_images, real_images)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Print progress every 10 batches
                if i % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{n_epochs}], Batch [{i}/{len(self.dataloader)}], '
                          f'Loss: {loss.item():.4f}')

            # Print epoch statistics
            avg_loss = total_loss / len(self.dataloader)
            print(f'Epoch [{epoch + 1}/{n_epochs}], Average Loss: {avg_loss:.4f}')

            # Save model every 5 epochs
            if (epoch + 1) % 5 == 0:
                print(f"Saving model at epoch {epoch + 1}...")
                torch.save(self.netG, self.filename)

        # Save final model
        print("Training completed. Saving final model...")
        torch.save(self.netG, self.filename)

    def generate(self, ske):
        """Generate an image from a skeleton"""
        # Get the device the model is on
        device = next(self.netG.parameters()).device

        # Prepare skeleton input
        ske_t = self.dataset.preprocessSkeleton(ske)
        ske_t_batch = ske_t.unsqueeze(0)  # Add batch dimension

        # Move input to the same device as model
        ske_t_batch = ske_t_batch.to(device)

        # Set model to evaluation mode
        self.netG.eval()

        with torch.no_grad():  # No need to track gradients during generation
            # Generate image
            normalized_output = self.netG(ske_t_batch)

            # Move output back to CPU for image processing
            normalized_output = normalized_output.cpu()

            # Convert to image format
            generated_image = self.dataset.tensor2image(normalized_output[0])

        return generated_image




if __name__ == '__main__':
    force = False
    optSkeOrImage = 2         # use as input a skeleton (1) or an image with a skeleton drawed (2)
    n_epoch = 200  # 200
    train = True #False
    #train = True

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenVanillaNN: Current Working Directory=", os.getcwd())
    print("GenVanillaNN: Filename=", filename)
    print("GenVanillaNN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)
    print("targetVideoSke",targetVideoSke)
    if train:
        # Train
        gen = GenVanillaNN(targetVideoSke, loadFromFile=False,optSkeOrImage=2)
        gen.train(n_epoch)
    else:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=False)    # load from file


    # Test with a second video
    for i in range(targetVideoSke.skeCount()):
        image = gen.generate( targetVideoSke.ske[i] )
        #image = image*255
        nouvelle_taille = (256, 256)
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)
