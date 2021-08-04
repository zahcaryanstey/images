import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from CustomDatasetv4 import CellDataset
import matplotlib.pyplot as plt
import rasterio
import numpy as np
import wandb
wandb.login()
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
"""
Step 2 set device 
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device used : ', device)


"""
Step 3 Define Hyper parameters 
"""
# Hyper parameters
config = dict(
    learning_rate=1e-3,
    batch_size=16,
    num_epochs=1,
    num_workers=1,
)
# print(config) # Prints the config dictionary as expected
# batch_size = 64
# num_workers = 1
# learning_rate = 1e-3
# num_epochs = 300
# epochs = num_epochs


"""
Paths to the data 
Csv = entire data set 
Ch1 = path to images 
train = path to training data set 
validation = path to test data set 
"""
csv = '/home/zachary/Desktop/DeepLearning/PreProcessing/HT29/HT29.csv'   # Path to csv file with labels and file names
Ch1 = '/home/zachary/Desktop/DeepLearning/Dataset/HT29/All/Ch1'   # Path to Ch1 images.
train = '/home/zachary/Desktop/DeepLearning/PreProcessing/HT29/HT29train.csv'
validation = '/home/zachary/Desktop/DeepLearning/PreProcessing/HT29/HT29test.csv'
# print(csv,Ch1,train,validation) # Prints the paths for the data as expected
loss_list = []
validation_loss_list = []
transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
# print(transform) # Prints the transforms as expected

# dataset = CellDataset(csv_file=csv, root_dir=Ch1, transform=transform)
train_set = CellDataset(csv_file=train, root_dir=Ch1, transform=transform)
validation_set = CellDataset(csv_file=validation, root_dir=Ch1, transform=transform)
# train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True,num_workers=num_workers)
# validation_loader = DataLoader(validation_set, batch_size=batch_size,shuffle=False,num_workers=num_workers)
# print(dataset, train_set, validation_set,train_loader,validation_loader) # Prints the datasets as expected

"""
Step 5 Mode Pipeline 


"""


def model_pipeline(hyperparameters):
    # Tell wandb to get started
    with wandb.init(project='test', config=hyperparameters):
        # acess all HPs through wandb.config, so logging matches execution
        config = wandb.config
        # print(config) # Prints the config dict defined above
        # make the model,data and optimization problem
        model, train_loader, validation_loader, criterion, optimizer=make(config)
        # and use them to train the model
        wandb.watch(model, criterion, log='all', log_freq=10)
        for epoch in range(config.num_epochs):

            train(model, train_loader, criterion, optimizer, config)
            # and test its final performance
            validation(model, validation_loader, criterion)
    return model


"""
Step 6 Make the data, model , loss and optimizer 
"""


def make(config):
    # Make the data
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    validation_loader = DataLoader(validation_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Make the model
    model = torchvision.models.resnet18(pretrained=True)
    # model.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
    model.conv1 = nn.Conv2d(1, 64, (7, 2), padding=3, bias=False)
    model.fc = nn.Linear(in_features=512, out_features=1)
    # print(model) # Prints the layers of the model as expected
    model.to(device)

    # Make the loss and optimizer
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    return model, train_loader, validation_loader, criterion, optimizer

"""
Step 7 Define the data loading and model 
"""
def get_data():
    full_dataset = CellDataset(csv_file=csv, root_dir=Ch1, transform=transform)

    return full_dataset
def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    return loader


"""
Step 8 Define Training logic 
"""
"""

1) track training loss per sample
    - In trainging loop  wandb.lop({'train_per_sample_loss' : train_loss})
    
"""
def train(model, train_loader, criterion, optimizer, config):
    # tell wandb to watch what the model gets ups to: gradients, weights and more
    # wandb.watch(model, criterion, log='all', log_freq=10)
    # Run training and track with wandb
    total_batches = len(train_loader) * config.num_epochs
    example_ct = 0
    batch_ct = 0

    for batch_idx, (data, targets) in enumerate(train_loader):
       #  print(data) # Prints image tensor as expected
       #  print(targets)
        loss = train_batch(data, targets, model, optimizer, criterion)
        # print(loss) # Prints loss as expected
        example_ct += len(data)
        batch_ct += 1
        # report metrics every 25th batch
        loss_list.append(loss.item())
        if((batch_ct + 1 ) % 5) == 0:
            train_log(loss, example_ct, config.num_epochs)
            #break

def train_batch(data, targets, model, optimizer, criterion):
    data, targets = data.to(device), targets.to(device)

    # forward pass
    scores = model(data)
    loss = criterion(scores, targets)
    # outputs = model(img)

    # loss = criterion(outputs,y_label)
    # backward pass
    optimizer.zero_grad()
    loss.backward()
    # step with optimizer
    optimizer.step()
    return loss
def train_log(loss, example_ct, epoch):
    train_loss = float(loss)


    wandb.log({'train_per_sample_loss': train_loss})
    wandb.log({'train_average_loss': np.mean(loss_list)})
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


def validation(model, validation_loader, criterion):
    valid_loss = 0
    model.eval()
    print('Validating model ')
# Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        for data, targets in validation_loader:

            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (targets).sum().item()
            # print(predicted)
            # print(targets)
        wandb.log({'Validation_average_loss': np.mean(valid_loss)})
        print(f"Accuracy of the model on the{total}" + f"test images:{100*correct/total}%")
        wandb.log({"test_accuracy": correct / total})


model = model_pipeline(config)
