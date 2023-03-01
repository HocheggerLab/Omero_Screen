import os
import ast
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from natsort import natsorted
from torchvision.utils import save_image
import torchvision.transforms as transforms
import skimage
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2




import glob

class CCdata(Dataset):
    def __init__(self, D_PATH,transform):
        self.X,self.Y=self.fetch_data(D_PATH)
        self.transform=transform


    def fetch_data(self,D_PATH):
        data_file = sorted(glob.glob(f'{D_PATH}/*.npy'))
        training_dicts = []
        for file in data_file:
            training_files = np.load(file, allow_pickle=True)
            training_dicts.append(training_files[()])
            images = []
            targets = []
            for dict_ in training_dicts:
                for j in range(len(dict_['target'])):
                    if dict_['data'][j].shape == (41, 41, 3):
                        images.append((dict_['data'][j]))
                        targets.append(dict_['target'][j])
        if len(images) == len(targets):
            X = np.array(images)
            #
            #     # X= X[:,:,:,np.newaxis]
            #     X = np.concatenate((X, X, X), axis=-1)
            Y = np.array(targets)
        return X,Y


    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x=self.X[index]
        y=self.Y[index]



        # y_lable = torch.tensor(np.array(ast.literal_eval(self.annotations.iloc[index,1]), dtype=np.float32))




        # print((np.float32(image) / 255)
        if self.transform:
            # x = self.transform(image=x)
            augmentations= self.transform(image=x)
            x=augmentations['image']

            # mean, std = torch.mean(image), torch.std(image)
            # transform_norm=transforms.Normalize(mean, std)
            # img_normalized = transform_norm(image)


        return x,y

if __name__=='__main__':
    transform = A.Compose(
        [
            A.Resize(32, 32, ),
            # # A.CenterCrop(60, 60, ),
            # # A.Resize(32, 32),
            A.Rotate(limit=20, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.VerticalFlip(p=0.6),
            A.VerticalFlip(p=0.4),
            A.OneOf(
                [
                    # A.Blur(blur_limit=3, p=0.8),
                    # A.Blur(blur_limit=3, p=0.5),
                    # A.ColorJitter(p=0.6),

                ], p=1.0
            ),
            ToTensorV2(),

        ]
    )
    dataset =CCdata(D_PATH='../CNN_pytorch/data/',transform=transform)
    # print(dataset.__getitem__(0)[0].permute(2, 0, 1))
    plt.imshow(dataset.__getitem__(0)[0].permute(2, 1, 0))
    plt.show()
