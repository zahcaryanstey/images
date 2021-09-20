import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CellDataset(Dataset):
    def __init__(self,root_dir,csv_file,transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform




    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, index):
        img_list = []
        for path in self.root_dir:
              img_path = os.path.join(path,self.annotations.iloc[index,0])
              image = rasterio.open(img_path).read().squeeze(0)
              if self.transform is not None:
                  img = self.transform(image)
              img_list.append(img)
        img = torch.cat(img_list,dim=0)
        y_label = torch.tensor(np.float(self.annotations.iloc[index,1]))
        return (img,y_label) # return images and labels
