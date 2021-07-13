from PIL import Image
from torchvision.transforms import ToTensor
import rasterio
import matplotlib.pyplot as plt
import  torchvision.transforms as transforms
path = '/home/zachary/Desktop/DeepLearning/Test_images/Ch1/27.tif'
image = rasterio.open(path).read().squeeze(0)
plt.imshow(image,cmap='gray')
plt.show()

image = ToTensor()(image)
print(image)
print('Type of image : ',type(image))
im = transforms.ToPILImage()(image).convert('RGB')
print('Type of im',type(im))
im.convert('RGB')
img = ToTensor()(im)
transforms.Resize(256)(img)
transforms.CenterCrop(10)(img)
print(img)
print(len(img))
im.show()
