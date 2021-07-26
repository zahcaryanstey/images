import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
class CellDataset(Dataset):
    def __init__(self,root_dir,csv_file,transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir,self.annotations.iloc[index,0])
        image = rasterio.open(img_path).read().squeeze(0)
        # y_label=float(self.annotations.loc[index,'NC'])
        y_label = torch.tensor(np.float(self.annotations.iloc[index,2]))

        # print(self.annotations.loc[index,'NC'])
        if self.transform is not None:
            img = self.transform(image)
        return(img,y_label)
