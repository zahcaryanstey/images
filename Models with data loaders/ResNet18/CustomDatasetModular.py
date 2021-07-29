import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio
from PIL import Image
import  torchvision.transforms as transforms
from torchvision.transforms import ToTensor
class CellDataset(Dataset):
    def __init__(self,csv_file,root_dir,ChannelList,transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.ChannelList = ChannelList

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        img_path = os.path.join(self.root_dir,self.annotations.iloc[index,4])
        for i, image in enumerate(channelList):
            print(i)
            image = rasterio.open(img_path).read().squeeze(0)
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224,224)),
                transforms.CenterCrop(10),
                transforms.ToTensor()
            ])
            image = transform(image)
            print('Channel {}'.format(i+1))



        # transforms.Resize((256,256))(img)
        # transforms.CenterCrop(10)(img)




        y_label = torch.tensor(int(self.annotations.iloc[index,3]))

        if self.transform:
            image = self.transform(image)
        return(image,y_label )



