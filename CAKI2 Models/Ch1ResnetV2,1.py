import torch
import torch.nn as nn
import torch.optim as optim
import  torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from CustomDatasetv2 import CellDataset
import math
import wandb
wandb.login()



# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device used :',device)

# Hyper parameters
config = dict(
    learning_rate=1e-3,
    batch_size=64,

    num_epochs=300,
    num_workers=1,
)
batch_size=32
num_workers = 1
learning_rate=1e-3
num_epochs = 100
epochs = num_epochs
# Load data
csv = '/home/zachary/Desktop/DeepLearning/PreProcessing/CAKI2/CAKI2.csv' # Path to csv file with labels and file names
Ch1 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/All/Ch1' # Path to Ch1 images.
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


def model_pipeline(hyperparameters):
    with wandb.init(project='CAKI2', config=hyperparameters):
        config = wandb.config
        model,train_loader,test_loader,criterion,optimizer=make(config)
        print(model)
        train(model,train_loader,criterion,optimizer,config)
        test(model,test_loader)

    return model

def make(config):

    train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True,num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False,num_workers=num_workers)
    model = torchvision.models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
    model.fc = nn.Linear(in_features=512,out_features=1)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    return model,train_loader,test_loader,criterion,optimizer
def get_data():
    full_dataset = CellDataset(csv_file=csv,root_dir=Ch1,transform=transform)
    return full_dataset
def make_loader(dataset,batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=num_workers)
    return loader
def train(model,loader,criterion,optimizer,config):
    wandb.watch(model,criterion,log='all',log_freq=10)
    total_batches = len(loader) * config.num_epochs
    example_ct = 0
    batch_ct = 0
    for epoch in range(config.num_epochs):
        for batch_idx,(data,targets) in enumerate(loader):
            loss = train_batch(data,targets,model,optimizer,criterion)
            example_ct += len(data)
            batch_ct += 1
            if((batch_ct + 1 ) % 25) == 0:
                train_log(loss,example_ct,epoch)
def train_batch(data,targets,model,optimizer,criterion):
    data, targets = data.to(device), targets.to(device)
    outputs = model(data)
    loss = criterion(outputs,targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
def train_log(loss,example_ct,epoch):
    loss = float(loss)
    wandb.log({'epoch':epoch,'loss':loss},step=example_ct)
    print(f"loss after"+str(example_ct).zfill(5)+f"examples:{loss:.3f}")

def test(model,test_loader):
    model.eval()
    with torch.no_grad():
        correct,total = 0,0
        for data,targets in test_loader:
            data,targets = data.to(device), targets.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data,1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        print(f"Accuracy of the model on the{total}" + f"test images:{100*correct/total}%")
        wandb.log({'test_accuracy', correct/total})

model = model_pipeline(config)





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
        data = data.to(device)
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




# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('device used :',device)
#
#
# csv = '/home/zachary/Desktop/DeepLearning/PreProcessing/CAKI2/CAKI2.csv' # Path to csv file with labels and file names
# Ch1 = '/hoCme/zachary/Desktop/DeepLearning/Dataset/CAKI2/All/Ch1' # Path to Ch1 images.
# train = '/home/zachary/Desktop/DeepLearning/PreProcessing/CAKI2/train/train.csv'
# test = '/home/zachary/Desktop/DeepLearning/PreProcessing/CAKI2/test/test.csv'
# transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((256,256)),
#             transforms.CenterCrop(224),
#             transforms.ToTensor()
#         ])
#
#
# config = dict(
#     epochs = 5,
#     batch_size = 64,
#     learning_rate = 1e-3,
#     dataset = CellDataset(csv_file=csv,root_dir=Ch1,transform=transform),
#     architecture = 'resnet18'
# )
#
#
#
#
# dataset = CellDataset(csv_file=csv,root_dir=Ch1,transform=transform)
#
# train_set = CellDataset(csv_file = train,root_dir = Ch1,transform=transform)
# test_set = CellDataset(csv_file = test,root_dir = Ch1,transform=transform)
#
# train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True,num_workers=num_workers)
# test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False,num_workers=num_workers)
#
#
#
#
#  # Model
# model = torchvision.models.resnet18(pretrained=True)
# model.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
# model.fc = nn.Linear(in_features=512,out_features=1)
# # Sigmoid activation function
# model.to(device)
#
# def model_pipeline(hyperparameters):
#     with wandb.init(project='test',config=hyperparameters):
#         config = wandb.config
#         model,train_loader,test_loader,criterion,optimizer=make(config)
#         print(model)
#         train(model,train_loader,criterion,optimizer,config)
#         test(model,test_loader)
#
#     return model
# def make(config):
#     train,test = get_data(train=True),get_data(train=False)
#     train_set = CellDataset(csv_file = train,root_dir = Ch1,transform=transform)
#     test_set = CellDataset(csv_file = test,root_dir = Ch1,transform=transform)
#     train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True,num_workers=num_workers)
#     test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False,num_workers=num_workers)
#     model = torchvision.models.resnet18(pretrained=True)
#     model.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
#     model.fc = nn.Linear(in_features=512,out_features=1)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
#     return model, train_loader, test_loader, criterion, optimizer
#
# def get_data(slice=5,train=True):
#     full_dataset = CellDataset(csv_file=csv,root_dir=Ch1,transform=transform)
#     return full_dataset
#
# def train(model,loader,criterion,optimizer,config):
#     wandb.watch(model,criterion,log='all',log_freq=10)
#     total_batches = len(loader) * config.epochs
#     example_ct = 0
#     batch_ct = 0
#     for epoch in range(config.epochs):
#         for _,(images,labels) in enumerate(loader):
#             loss = train_batch(images,labels,model,optimizer,criterion)
#             example_ct += len(images)
#             batch_ct += 1
#             if ((batch_ct + 1) % 25) == 0:
#                 train_log(loss,example_ct,epoch)
#
# def train_batch(images, labels, model, optimizer, criterion):
#     images, labels = images.to(device), labels.to(device)
#     outputs = model(images)
#     loss =criterion(outputs,labels)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     return loss
#
# def train_loss(loss, example_ct,epoch):
#     loss = float(loss)
#     wandb.log({'epoch':epoch,'loss':loss}, step = example_ct)
#     print(f"loss" + str(example_ct).zfill(5) + f"examples {loss:.3f}")
#
#
# def test(model, test_loader):
#     model.eval()
#
#     # Run the model on some test examples
#     with torch.no_grad():
#         correct, total = 0, 0
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#         print(f"Accuracy of the model on the {total} " +
#               f"test images: {100 * correct / total}%")
#
#         wandb.log({"test_accuracy": correct / total})
#
#     # Save the model in the exchangeable ONNX format
#     torch.onnx.export(model, images, "model.onnx")
#     wandb.save("model.onnx")
# model = model_pipeline(config)
