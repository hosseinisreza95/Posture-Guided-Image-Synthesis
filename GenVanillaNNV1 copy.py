import numpy as np
import cv2
import os
import pickle
import sys
import math

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.io import read_image
import torch.optim as optim


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




def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)




class GenNNSkeToImage(nn.Module):
    def __init__(self):
        super(GenNNSkeToImage, self).__init__()
        self.input_dim = Skeleton.reduced_dim  # همانطور که گفته شده ۲۶ بعدی
        self.model = nn.Sequential(
            nn.Linear(26, 3 * 64 * 64),
            nn.ReLU(),
            nn.Linear(3 * 64 * 64, 3 * 64 * 64),
            nn.Tanh()  # برای اینکه خروجی‌ها در محدوده [-1, 1] باشد
        )
        print(self.model)

    def forward(self, z):
        img = self.model(z)
        img = img.view(-1, 3, 64, 64)  # شکل‌دهی مجدد به خروجی
        return img





class GenNNSkeImToImage(nn.Module):
    """ class that Generate a new image from from THE IMAGE OF the new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self):
        super(GenNNSkeToImage, self).__init__()
        self.input_dim = Skeleton.reduced_dim
        self.model = nn.Sequential(
            # TP-TODO
        )
        print(self.model)

    def forward(self, z):
        img = self.model(z)
        return img



class GenVanillaNN():
    """ class that Generate a new image from a new skeleton posture
        Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        image_size = 64
        if optSkeOrImage == 1:
            self.netG = GenNNSkeToImage()
            src_transform = None
            self.filename = 'data/Dance/DanceGenVanillaFromSke.pth'
        else:
            self.netG = GenNNSkeImToImage()
            src_transform = transforms.Compose([ 
                SkeToImageTransform(image_size),
                transforms.ToTensor(),
            ])
            self.filename = 'data/Dance/DanceGenVanillaFromSkeim.pth'

        tgt_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=src_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=16, shuffle=True)

        # تعریف optimizer و criterion
        self.criterion = nn.MSELoss()  # می‌توانید از معیار خطاهای مختلف استفاده کنید
        self.optimizer = optim.Adam(self.netG.parameters(), lr=0.001)  # تنظیم optimizer برای مدل

        if loadFromFile and os.path.isfile(self.filename):
            print("GenVanillaNN: Load=", self.filename)
            print("GenVanillaNN: Current Working Directory: ", os.getcwd())
            self.netG = torch.load(self.filename)

    def train(self, n_epochs=20):
        # حلقه‌ی اصلی آموزش
        for epoch in range(n_epochs):
            epoch_loss = 0.0

            # حلقه‌ی داخلی برای هر دسته از داده‌ها (batch)
            for ske, image in self.dataloader:
                # تغییر شکل `ske` به `(batch_size, 26)` برای سازگاری با مدل
                ske = ske.view(-1, 26)

                # صفر کردن گرادیان‌ها
                self.optimizer.zero_grad()

                # پیش‌بینی با استفاده از مدل
                output = self.netG(ske)

                # محاسبه‌ی خطا
                loss = self.criterion(output, image)
                epoch_loss += loss.item()

                # محاسبه و اعمال گرادیان‌ها
                loss.backward()
                self.optimizer.step()

            # چاپ خطا برای هر دوره
            print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss / len(self.dataloader)}')

            # ذخیره مدل پس از هر دوره
            torch.save(self.netG, self.filename)
            print(f'Model saved to {self.filename}')

        # در انتهای آموزش مدل ذخیره می‌شود
        print("Training complete. Saving model...")
        torch.save(self.netG, self.filename)
        print(f'Final model saved to {self.filename}')


    def generate(self, ske):
        """ generator of image from skeleton """
        ske_t = self.dataset.preprocessSkeleton(ske)
        ske_t_batch = ske_t.unsqueeze(0)        # make a batch
        with torch.no_grad():
            normalized_output = self.netG(ske_t_batch)
        res = self.dataset.tensor2image(normalized_output[0])       # get image 0 from the batch
        return res





if __name__ == '__main__':
    force = False
    optSkeOrImage = 1           # use as input a skeleton (1) or an image with a skeleton drawed (2)
    n_epoch = 1000  # 200
    # train = 1 #False
    train = True

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

    if train:
        # Train
        gen = GenVanillaNN(targetVideoSke, loadFromFile=False)
        gen.train(n_epoch)
    else:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=True)    # load from file        


    # Test with a second video
    for i in range(targetVideoSke.skeCount()):
        image = gen.generate( targetVideoSke.ske[i] )
        #image = image*255
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)
