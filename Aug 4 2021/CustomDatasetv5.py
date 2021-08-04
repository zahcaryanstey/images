import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import matplotlib.pyplot as plt
class CellDataset(Dataset):
    def __init__(self,root_dir,csv_file,transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform
        # print(self.annotations.head()) # prints the first 5 row of the each data set as expected



    def __len__(self):
        # print(len(self.annotations)) # prints the length of each data set as expected
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir,self.annotations.iloc[index,0])
        # print(img_path) #prints the path to the image as extected

        image = rasterio.open(img_path).read().squeeze(0)
        # plt.imshow(image,cmap='gray')
        # plt.show()
        # print(image) # Prints image pixel values as expected
        # y_label=float(self.annotations.loc[index,'NC'])

        y_label = torch.tensor(np.float(self.annotations.iloc[index,1]))

        # print(y_label) # Prints ground truth label as expected



        # print(self.annotations.loc[index,'NC'])
        if self.transform is not None:
            img = self.transform(image)
            # print(img,y_label) # prints both the image and the ground truth label as expected

        return(img,y_label)
