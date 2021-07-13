import torch
import torch.nn as nn
import torch.optim as optim
import  torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from CustomDataset import CellDataset

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device used :',device)

# Hyper parameters
learning_rate = 1e-3
batch_size = 32
num_epochs = 1

# Load data
csv = '/home/zachary/Desktop/DeepLearning/Pre processing/CAKI2.csv'
# Ch1 = '/home/zachary/Desktop/DeepLearning/Test_images/Ch1'
Ch1 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/All/Ch1'
# dataset = CellDataset(csv_file= csv, root_dir = Ch1,transform = transforms.Resize(224) )
dataset = CellDataset(csv_file=csv,root_dir=Ch1,transform=transforms.ToTensor())
train_set, test_set = torch.utils.data.random_split(dataset, [877,219])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size,shuffle=True)

# Model
model = torchvision.models.resnet18(pretrained=True)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

# Train the network
for epoch in range(num_epochs):
    losses = []
    for batch_idx,(data,targets) in enumerate(train_loader):
        # get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device)

        # Forward
        scores = model(data)
        loss = criterion(scores,targets)
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f'cost at epoch{epoch} is {sum(losses)/len(losses)}')

# Check accuracy on training to see how good our model is
def check_accuracy(loader,model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _,predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 1}')
    model.train()
print('Checking accuracy on Training set ')
check_accuracy(train_loader,model)

print('Cecking accuracy on test set ')
check_accuracy(test_set,model)
