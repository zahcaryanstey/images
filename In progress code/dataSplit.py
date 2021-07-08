"""
This is a script to split the data set into testing, training, and validation sets
The test split we will use is:
70 % training
15 % testing
15 % validation
"""

import os
import numpy as np
import shutil
import random

# Creating Traing / Val / Test folders
root_dir = '/home/zachary/Desktop/DeepLearning/Test_images/cell_images/Test_Image_Split'

# Ch1 = '/home/zachary/Desktop/DeepLearning/Test_images/cell_images/Test_Image_Split/Ch1'
# Ch2 = '/home/zachary/Desktop/DeepLearning/Test_images/cell_images/Test_Image_Split/Ch2'
# Ch3 = '/home/zachary/Desktop/DeepLearning/Test_images/cell_images/Test_Image_Split/Ch3'
# Ch4 = '/home/zachary/Desktop/DeepLearning/Test_images/cell_images/Test_Image_Split/Ch4'
# Ch5 = '/home/zachary/Desktop/DeepLearning/Test_images/cell_images/Test_Image_Split/Ch5'
# Ch6 = '/home/zachary/Desktop/DeepLearning/Test_images/cell_images/Test_Image_Split/Ch6'
# Ch7 = '/home/zachary/Desktop/DeepLearning/Test_images/cell_images/Test_Image_Split/Ch7'
# Ch8 = '/home/zachary/Desktop/DeepLearning/Test_images/cell_images/Test_Image_Split/Ch8'
# Ch9 = '/home/zachary/Desktop/DeepLearning/Test_images/cell_images/Test_Image_Split/Ch9'
# Ch10 = '/home/zachary/Desktop/DeepLearning/Test_images/cell_images/Test_Image_Split/Ch10'
# Ch11 = '/home/zachary/Desktop/DeepLearning/Test_images/cell_images/Test_Image_Split/Ch11'
# Ch12 = '/home/zachary/Desktop/DeepLearning/Test_images/cell_images/Test_Image_Split/Ch12'

classes_dir = ['/Ch1','/Ch2','/Ch3','/Ch4','/Ch5','/Ch6','/Ch7','/Ch8','/Ch9','/Ch10','/Ch11','/Ch12']

val_ratio = 0.25
test_ratio = 0.1

for cls in classes_dir:
    os.makedirs(root_dir + '/train' + cls)
    os.makedirs(root_dir + '/val' + cls)
    os.makedirs(root_dir + '/test' + cls)

    # creating partitions of the data after shuffeling
    src = root_dir + cls # folder to copy images from

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames)*(1-val_ratio + test_ratio)),
                                                               int(len(allFileNames)*(1 - test_ratio))])
    train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
    val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src+'/'+name for name in test_FileNames.tolist()]

    print('Total images:',len(allFileNames))
    print('Trainging:',len(train_FileNames))
    print('Validation:',len(val_FileNames))
    print('Testing:',len(test_FileNames))

    # Copy pasting images
    for name in train_FileNames:
        shutil.copy(name,root_dir+'/train'+cls)
    for name in val_FileNames:
        shutil.copy(name,root_dir+'/val'+cls)
    for name in test_FileNames:
        shutil.copy(name, root_dir +'/test'+cls)


