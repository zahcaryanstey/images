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
        img_list = []
        for path in self.root_dir:
              img_path = os.path.join(self.root_dir,self.annotations.iloc[index,0])
              image = rasterio.open(img_path).read().squeeze(0)
              img_list.append(image)
              image = np.stack(img_list,axis=0)
         # img_path = os.path.join(self.root_dir,self.annotations.iloc[index,0]) d
         # image = rasterio.open(img_path).read().squeeze(0) # read the images
              y_label = torch.tensor(np.float(self.annotations.iloc[index,1]))
              if self.transform is not None:
                  img = self.transform(image)
             # print(img,y_label) # prints both the image and the ground truth label as expected

                  return(img,y_label) # return images and labels
