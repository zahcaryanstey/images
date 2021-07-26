import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio
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
        y_label = torch.tensor(float(self.annotations.iloc[index,3]))
        print(self.annotations.iloc[index,4])
        if self.transform is not None:
            img = self.transform(image)
        return(img,y_label)
