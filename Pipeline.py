# Training pipeline




"""
TEST WITH ONE CHANNEL [CH1] From the CAKI2 Images.
"""

"""
Algorithm for training pipeline
1) import libaries
2) Load data
3) Visualize a few images
4) train the model
5) visualize the model predicions
6) fine tune the convenutional network
7) train and evaluate
8) convnet as fixed feature extractor
9) train and evaluate
"""

import matplotlib.pyplot as plt
import torch.cuda
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms,models

import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor
import rasterio
import torchvision.models as models
"""
WHAT WE NEED 
1) Load cell images from data loader 
2) Define a convolutional neural network
3) Define a loss function
4) Define a optimizer 
5) Train the network on the training data 
6) Validation 
7) Test the network on the test data (last thing to do ) 
"""




# Defnine the train/validation data set loader, using the SubsetRandomSampler for the split
data_dir = '/home/zachary/Desktop/DeepLearning/Test_images/cell_images/Test_Image_Split/Ch1'

def load_split_train_test(datadir, valid_size = .2):
    train_transforms = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),])
    test_transforms = transforms.Compose([transforms.Resize(224),transforms.ToTensor(),])

    train_data = datasets.ImageFolder(datadir,transform=train_transforms)
    test_data = datasets.ImageFolder(datadir,transform=test_transforms)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size = 64)
    return trainloader, testloader
trainloader, testloader = load_split_train_test(data_dir,.2)
print(trainloader.dataset.classes)




# ChannelList = [1,7,11]


# define our resnet18 model

resnet18 = models.resnet18(pretrained=True,progress=True) # pretrained resnet model.

# Find if a GPU is available and use it if not use cpu
device = torch.device('cuda' if torch.cuda.is_available() else'cpu')
print(device)

"""
Now we have to stop the pre trained layers, so that we do not backpropagate through them during training.
Next we redefine the final fully connected layer ( the one we will use with our images)
Also, we will define a loss function and pick an optimizer 
"""
for param in resnet18.parameters():
    param.requries_grad = False

resnet18.fc = nn.Sequential(nn.Linear(2048,512),nn.ReLU(),nn.Dropout(0.2),nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(resnet18.fc.parameters(),lr = 0.003)
resnet18.to(device)

"""
Now we can train our model. The basic process is quite intuitive from the code:
- You define the number of epochs 
- You load the batches of images 
- Do the feed forward loop
- Calculate the loss function 
- use optimizer to apply gradient descent in back propagation 
- During validation, do not forget to set the model to eval() mode, and then back to train once you are done 
"""

epochs = 1
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [],[]

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps +=1
        inputs, labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        logps = resnet18.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            resnet18.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device),labels.to(device)
                    logps = resnet18.model.forward(inputs)
                    batch_loss = criterion(logps,labels)
                    test_loss += batch_loss.item()
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1,dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))
            print(f"Epoch {epoch}/{epochs}..."
                  f"Train loss: {running_loss/print_every:.3f} .."
                  f"Test loss: {test_loss/len(testloader):.3f}.."
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            resnet18.train()
torch.save(model,'aerialmodel.pth')