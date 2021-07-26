import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio
from PIL import Image
import  torchvision.transforms as transforms
from torchvision.transforms import ToTensor
class CellDataset(Dataset):
    def __init__(self,csv_file,root_dir,transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        img_path = os.path.join(self.root_dir,self.annotations.iloc[index,0])
        image = rasterio.open(img_path).read().squeeze(0)







        # transforms.Resize((256,256))(img)
        # transforms.CenterCrop(10)(img)




        y_label = torch.tensor(int(self.annotations.iloc[index,4]))

        if self.transform:
            image = self.transform(image)
        return(image,y_label )



