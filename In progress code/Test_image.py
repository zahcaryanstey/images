# Script to open tiff images.
#  Note: Original images have a file format of .ome.tif
# For this procedure to work we need to save the files as .tiff format

# The libaries required for this to work are
# Pillow
# torchvision
# raterio
# Matplotlib


# First we need to import the necessary libraries
from PIL import Image
from torchvision.transforms import ToTensor
import rasterio
import matplotlib.pyplot as plt



# now use rasterio to open images
path = '/home/zachary/Desktop/Test_images/cell_images/27_Ch1.tif'
image1 = rasterio.open(path).read().squeeze(0)

# Next display the images
plt.imshow(image1, cmap='gray')
plt.show()

# Finnaly convert the images to tensors
image1 = ToTensor()(image1)
print(image1)