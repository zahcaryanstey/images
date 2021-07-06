
# This is a script to change the file extension on our images
# The original images have a file extension of .ome.tif
# We need to change the extension so that the images
# are now saved as .tiff





# First we need to import the necessary libraries
import os



folder = '/home/zachary/Desktop/Masters Degree /Test_images/cell_images'
import os

paths = (os.path.join(root, filename)
        for root, _, filenames in os.walk(folder)
        for filename in filenames)

for path in paths:
    # take the .ome.tif part of the file name and change to .tif
    newname = path.replace('_Ch1','')
    if newname != path:
        os.rename(path, newname)