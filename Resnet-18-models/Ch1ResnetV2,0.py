import torch
import torch.nn as nn
import torch.optim as optim
import  torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from CustomDatasetv2 import CellDataset
import math





# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device used :',device)

# Hyper parameters
learning_rate = 1e-3
batch_size = 32
num_epochs = 10
num_workers = 0
# Load data

csv = '/home/zachary/Desktop/DeepLearning/PreProcessing/CAKI2/CAKI2.csv' # Path to csv file with labels and file names
Ch1 = '/hoCme/zachary/Desktop/DeepLearning/Dataset/CAKI2/All/Ch1' # Path to Ch1 images.
train = '/home/zachary/Desktop/DeepLearning/PreProcessing/CAKI2/train/train.csv'
test = '/home/zachary/Desktop/DeepLearning/PreProcessing/CAKI2/test/test.csv'


transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256,256)),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
dataset = CellDataset(csv_file=csv,root_dir=Ch1,transform=transform)

train_set = CellDataset(csv_file = train,root_dir = Ch1,transform=transform)
test_set = CellDataset(csv_file = test,root_dir = Ch1,transform=transform)

train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True,num_workers=num_workers)
test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False,num_workers=num_workers)

 # Model
model = torchvision.models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
model.fc = nn.Linear(in_features=512,out_features=1)
# Sigmoid activation function
model.to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

# Train the network
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / batch_size)
print('Total number of samples ',total_samples,'Number of iterations ',n_iterations)


for epoch in range(num_epochs):
    losses = []
    for batch_idx,(data,targets) in enumerate(train_loader):
        # get data to cuda if possible
        data = data.to(device=device)
        targets = targets.type(torch.FloatTensor).to(device)
        if (batch_idx + 1) % 5 == 0:
            print(f'epoch{epoch+1}/{num_epochs},step{batch_idx + 1}/{n_iterations}')
        # Forward
        scores = model(data)
        loss = criterion(scores,targets)
        losses.append(loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward()
        # gradient descent or adam step
        optimizer.step()
# Check accuracy on training to see how good our model is
def check_accuracy(loader,model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for batch_idx,(x, y) in enumerate(loader):
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            _,predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f'got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 1}')
model.train()
# print('Checking accuracy on Training set ')
# check_accuracy(train_loader,model)
#
# print('Cecking accuracy on test set ')
# check_accuracy(test_set,model)


