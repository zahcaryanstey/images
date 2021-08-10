""" libraries"""
import os # os library used to get the path for the files
import pandas as pd # pandas used for loading the data
import torch # torch used to change the images to tensors
from torch.utils.data import Dataset # torch utils data set used to build our custom data set
import rasterio # rasterio used to open the images
import numpy as np # numpy used to change the annotations to floats
import matplotlib.pyplot as plt # Used for plotting
class CellDataset(Dataset): # Class used to define our custom data loader
    def __init__(self,root_dir,csv_file,transform=None):# This function will take as input the root directory (image directory), csv file (labeling and file names), and the transforms of the images
        self.root_dir = root_dir # root directory is the root directory defined in the model
        self.annotations = pd.read_csv(csv_file) # this loads the csv files using pandas
        self.transform = transform # pass the transforms on the images defined in model

    def __len__(self):
        return len(self.annotations) # returns the length of the dataset

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir,self.annotations.iloc[index,0]) # the path to the images is found from the annotations csv file defined above from the annoations csv the file names are found in the first column
        image = rasterio.open(img_path).read().squeeze(0) # read the images

        # Below is code the I used to visualize a few images and there ground truth
       # plt.imshow(image,cmap='gray')
       # plt.show()
       # plt.title(self.annotations.iloc[index,1])
       ####
        y_label = torch.tensor(np.float(self.annotations.iloc[index,1])) # y_label is the NC ratio from the dataset converted to a float and then a torch tensor
        if self.transform is not None: # Apply transformations to the images
            img = self.transform(image)
            # print(img,y_label) # prints both the image and the ground truth label as expected

        return(img,y_label) # return images and labels 
