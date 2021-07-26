"""
Custom data loader
Algorithm
1) Import libaries
2) open data table
3) Show sample image
    i) Image number
    ii) image nucleus diameter
    iii) image cell diameter
    iv) Image NC ratio
4) Show image function
    i) Displays an image from each channel
    ii) Converts image to Tensor
5) Data set class to read the data set and images
6) Initial the class and iterate through the data samples
7) rescale class to rescale the images
8) Class to crop the images using center crop
9) Show sample of rescaled and cropped image
10) Iterate through the data set.
"""


#1) Libaries
from __future__ import print_function, division
import os
import pandas as pd
from PIL import Image
from torchvision.transforms import ToTensor
import rasterio
import matplotlib.pyplot as plt
import torch
plt.ion()
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Fix for a bug that I was having
import torch
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
plt.ion()   # interactive mode

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
# print('done1')

# 2) Data table
# Here I am going to load the data table
data = pd.read_csv('CAKI2NC.csv')
# Print the frist 5 rows to make sure the data table looks ok
print(data.head())
# The columns of the data table are as follows
# Object_Number  Cell_Diameter   Nucleus_Diameter  NC

# print('done2')


# 3) Sample image
# print('done3')
n = 0
img_name = data.iloc[n,0] # Image name comes from the first column of the data table (Object_Number)
cell_diameter = data.iloc[n,1] # Cell diameter of the image comes from the second column of the data table (Cell_Diameter)
nucleus_diameter = data.iloc[n,2] # nucleus diameter of the image comes from the third comlumn of the data table (Nucleus_Diameter)
NC = data.iloc[n,3] # NC ratio of the cell comes from the third and final column of the data table (NC)
print('Image name:{}'.format(img_name))
print('Cell Diameter:{}'.format(cell_diameter))
print('NC ratio:{}'.format(NC))

print('*' * 60 )
# print('done4')
# 4) Show image function
"""
Since we have images from all 12 channels of the IFC IDEAS software we must create a list 
that contains the paths to all 12 channels so that when we call on an image we can obtain
an image from all 12 channels. We can then define a function to open the 12 images 
and convert them to a tensor all in one step. 
"""

# Before we define our channel list we first have to define all of our channels.




# print('Done5')
Ch1 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/Processed/Ch1/27.tif'
Ch2 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/Processed/Ch2/27.tif'
Ch3 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/Processed/Ch3/27.tif'
Ch4 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/Processed/Ch4/27.tif'
Ch5 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/Processed/Ch5/27.tif'
Ch6 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/Processed/Ch6/27.tif'
Ch7 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/Processed/Ch7/27.tif'
Ch8 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/Processed/Ch8/27.tif'
Ch9 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/Processed/Ch9/27.tif'
Ch10 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/Processed/Ch10/27.tif'
Ch11 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/Processed/Ch11/27.tif'
Ch12 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/Processed/Ch12/27.tif'

print('Done6')

# Notes
# Rename files to have no spaces or dots
# Try with 16 bit images
# Start trying to train a model on our data.
# Start building training pipeline for model.
# Optimizer L1 oor L2 loss
# Ideas acquistion file properties.
# Focus on channels that are used.
# Moduler code
####################################################################
##  Test print an image from each channel.                         #
#image1 = rasterio.open(Ch12).read().squeeze(0)                    #
#plt.imshow(image1, cmap='gray')                                   #
#plt.show()                                                        #
###################################################################

#4) show image function
channelList =[Ch1,Ch2,Ch3,Ch4,Ch5,Ch6,Ch7,Ch8,Ch9,Ch10,Ch11,Ch12]

# print('Done7')

def show_image(cell_diameter,nucleus_diameter,NC,channelList):
    # First we want to print the name of the image, the cell and nucleus diameter of the image as well as the nc ratio of the image
    print('Image name:{}'.format(img_name))
    print('Cell diameter:{}'.format(cell_diameter ))
    print('Nucleus diameter:{}'.format(nucleus_diameter))
    print('NC ratio:{}'.format(NC))
    # Now we need a for loop to loop through each channel in our channel list and convert them to tensors
    for i, image in enumerate(channelList):
        if i == 0 or i == 6 or i == 10:
            print(i)

             # open images
            im = rasterio.open(image).read().squeeze(0)
            im = Image.fromarray(im)

            plt.imshow(im,cmap='gray')
            plt.show()
            # change to tensor.
            transform = transforms.Compose(
                [ transforms.CenterCrop(10),
                  transforms.Resize(256),
                    #transforms.ToTensor(),
                 ])
            image = transform(im)




            # torchvision.transforms.CenterCrop(10)
            # torchvision.transforms.Resize(256)
            # im = ToTensor()(im)
          #  torch.stack([image],dim=0)
            # print(image)
            print('Channel {}'.format(i+1))
            print(np.unique(np.array(image)))
            #print(image.shape)
            # print('Done8')

show_image(cell_diameter,nucleus_diameter,NC,channelList)
# print('Done9')

# 5) Data set class
# torch.utils.data.Datset is an abstract class representing a dataset. Your custom dataset should inherit Dataset and override the following methods
#  __len__ so that len(dataset) returns the size of the dataset
#  __getitem__ to support the indexing such that dataset[i] can be used to get the ith sample.
# Let's create a dataset class for our cell dataset. We will read the csv in __init__ but leave the reading of images to __getitem__.
# This is memory efficient because all the images are not stored in the memory at once but read as required. Sample of our dataset will be a
# dict {'image':image,'cell diameter':cell diameter,'nucleus diameter':nucleus diameter, 'NC':NC}. Our dataset will take an optional argument transform
# so that any required processing can be applied on the sample. We will see te usefulness of transform in the next section

Ch1 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/Processed/Ch1/27.tif'
Ch2 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/Processed/Ch2/'
Ch3 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/Processed/Ch3/'
Ch4 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/Processed/Ch4/'
Ch5 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/Processed/Ch5/'
Ch6 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/Processed/Ch6/'
Ch7 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/Processed/Ch7/'
Ch8 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/Processed/Ch8/'
Ch9 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/Processed/Ch9/'
Ch10 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/Processed/Ch10/'
Ch11 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/Processed/Ch11/'
Ch12 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/Processed/Ch12/'

channelList = Ch1
class CellDataset(Dataset):
    # print('Done10')
    def __init__(self,csv_file,root_dir,transform=None):
        # print('Done11')
        self.dataset = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):
        # print('Done12')
        print(len(self.dataset))  # This should print how many images are in the dataset
        return len(self.dataset)
    def __getitem__(self,idx):
        # print('Done13')
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = (self.root_dir)#,self.dataset.iloc[idx,0])
        # image = io.imread(img_name)
        image = rasterio.open(img_name).read().squeeze(0)
        cell_diameter = self.dataset.iloc[idx,1]
        nucleus_diameter = self.dataset.iloc[idx,2]
        NC = self.dataset.iloc[idx,3]
        cell_diameter = np.array([cell_diameter])
        nucleus_diameter = np.array([nucleus_diameter])
        NC = np.array([NC])
        cell_diameter = cell_diameter.astype(int)
        nucleus_diameter = nucleus_diameter.astype(int)
        NC = NC.astype(int)
        sample = {'image':image,'cell_diameter':cell_diameter,'nucleus_diameter':nucleus_diameter,'NC':NC}
        if self.transform:
            sample = self.transform(sample)
        return sample


#  Let's instantiate this class and iterate through the data samples. We will print the sizes of first 4 samples and show there
    # cell diameter, nucleus diameter and NC
# print('Done14')
cell_dataset = CellDataset(csv_file='CAKI2NC.csv',root_dir=channelList)
# print('Done15')
plt.show()
for i in range(len(cell_dataset)):
    print(cell_dataset[i])
    # sample = cell_dataset[i]
    print('Code done ')
# Error somewhere between print 16 and End of code
for i in range(len(cell_dataset)):
    # print('Done16')



    sample = cell_dataset[i]


    print(i,sample['image'].shape['landmarks'].shape)


    ax = plt.subplot(1,4,i+1)
    plt.tight_layout
    ax.set_title('sample #{}'.format(i))
    ax.axis('off')
    show_image(**sample)
    if i == 3:
        plt.show()
        break
print('Error fixed')








