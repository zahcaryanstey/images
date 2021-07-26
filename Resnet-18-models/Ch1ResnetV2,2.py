"""
New in this version
1) track training loss per sample
    - In trainging loop  wandb.lop({'train_per_sample_loss' : train_loss})
2) track average training loss
    - After training loop but before validation
    - wanb.log({train_ave_loss':np.mean(train_loss)}
3) track validation average loss
    - loss_list is a list of all the loss values (i.e., loss_list.append(train_loss.item()) at each time step)
    - at the end of the validation loop do
    - wandb.log({‘val_ave_loss’: np.mean(val_loss_list)})
"""



"""
Step 1 import libraries
"""
import torch
import torch.nn as nn
import torch.optim as optim
import  torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from CustomDatasetv3 import CellDataset
import math
import numpy as np
import wandb
wandb.login()



"""
Step 2 set device 
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device used :',device)


"""
Step 3 Define Hyper parameters 
"""
# Hyper parameters
config = dict(
    learning_rate=1e-3,
    batch_size=64,
    num_epochs=10,
    num_workers=1,
)
batch_size=32
num_workers = 1
learning_rate=1e-3
num_epochs = 1
epochs = num_epochs


"""
 Step 4 Load data 
"""
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


"""
Step 5 Mode Pipeline 
"""
def model_pipeline(hyperparameters):
    # Tell wandb to get started
    with wandb.init(project='Ch1ResnetV2,2', config=hyperparameters):
        # acess all HPs through wandb.config, so logging matches execution
        config = wandb.config
        # make the model,data and optimization problem
        model,train_loader,test_loader,criterion,optimizer=make(config)
        # and use them to train the model
        train(model,train_loader,criterion,optimizer,config)
        # and test its final performance
        test(model,test_loader)

    return model

"""
Step 6 Make the data, model , loss and optimizer 
"""
def make(config):
    # Make the data
    train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True,num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False,num_workers=num_workers)

    # Make the model
    model = torchvision.models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
    model.fc = nn.Linear(in_features=512,out_features=1)

    model.to(device)

    # Make the loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    return model,train_loader,test_loader,criterion,optimizer

"""
Step 7 Define the data loading and model 
"""
def get_data():
    full_dataset = CellDataset(csv_file=csv,root_dir=Ch1,transform=transform)
    return full_dataset
def make_loader(dataset,batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,pin_memory=True,num_workers=num_workers)
    return loader


"""
Step 8 Define Training logic 
"""
"""

1) track training loss per sample
    - In trainging loop  wandb.lop({'train_per_sample_loss' : train_loss})
    
"""
def train(model,loader,criterion,optimizer,config):
    # wandb.watch(model,criterion,log='all',log_freq=10)
    # wandb.log({'train_per_sample_loss': loss})
    total_batches = len(loader) * config.num_epochs
    example_ct = 0
    batch_ct = 0
    # Run training and track with wandb
    for epoch in range(config.num_epochs):
        for batch_idx,(data,targets) in enumerate(loader):
            loss = train_batch(data,targets,model,optimizer,criterion)
            example_ct += len(data)
            batch_ct += 1
            # report metrics every 25th batch
            if((batch_ct + 1 ) % 25) == 0:
                train_log(loss,example_ct,epoch)
def train_batch(data,targets,model,optimizer,criterion):
    data, targets = data.to(device), targets.to(device)
    # forward pass
    outputs = model(data)
    loss = criterion(outputs,targets)
    # backward pass
    optimizer.zero_grad()
    loss.backward()
    # step with optimizer
    optimizer.step()
    return loss
def train_log(loss,example_ct,epoch):
    train_loss = float(loss)
    loss_list = []
    loss_list.append(train_loss)

    wandb.log({'train_per_sample_loss':train_loss})
    wandb.log({'train_average_loss':np.mean(loss_list)})
    print(f"loss after"+str(example_ct).zfill(5)+f"examples:{train_loss:.3f}")


"""
2) track average training loss
    - After training loop but before validation
    - wanb.log({train_ave_loss':np.mean(train)
"""


"""
3) track validation average loss
    - loss_list is a list of all the loss values (i.e., loss_list.append(train_loss.item()) at each time step)
    - at the end of the validation loop do
    - wandb.log({‘val_ave_loss’: np.mean(val_loss_list)})
"""

"""
Step 9 Define Testing Logic 
"""
def test(model,test_loader):
    model.eval()
    print('Validating model ')

# Run the model on some test examples

    with torch.no_grad():
        correct,total = 0,0
        for data,targets in test_loader:
            data,targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data,1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        print(f"Accuracy of the model on the{total}" + f"test images:{100*correct/total}%")
        wandb.log({"test_accuracy": correct / total})


model = model_pipeline(config)









