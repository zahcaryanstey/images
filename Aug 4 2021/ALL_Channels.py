import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from DataLoader_All_Channels import CellDataset
import numpy as np
import matplotlib.pyplot as plt
import wandb
wandb.login()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device being used :',device)


hyperparameter = dict(
    learning_rate = 1e-3,
    batch_size = 16,
    num_epochs = 10,
    num_workers = 1,
)

csv = '/home/zachary/Desktop/DeepLearning/PreProcessing/CAKI2/CAKI2.csv'
Ch1 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/All/Ch1'
Ch7 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/All/Ch7'
Ch11 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/All/Ch11'
train = '/home/zachary/Desktop/DeepLearning/PreProcessing/CAKI2/train/train.csv'
validation = '/home/zachary/Desktop/DeepLearning/PreProcessing/CAKI2/test/test.csv'
Channels = [Ch11]

transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
for i in range(len(Channels)):
    channel = Channels[i]
    train_set = CellDataset(csv_file=train, root_dir=channel, transform=transform)
    print('Training on:',channel)
    validation_set = CellDataset(csv_file=validation, root_dir=channel, transform=transform)
    print('Validating on',channel)

def model_pipeline(hyperparameters):
    with wandb.init(project='test',config=hyperparameters):
        hyperparameter = wandb.config
        model, train_loader, validation_loader,criterion,optimezer=make(hyperparameter)
        second_loss = nn.L1Loss()
        wandb.watch(model,criterion,log='all',log_freq=10)
        for epoch in range(hyperparameter.num_epochs):
            train(model,train_loader,criterion,optimezer,hyperparameter,second_loss)
            validation(model,validation_loader,criterion)
    return model

def make(hyperparameter): # function to make the data and the model
    train_loader = DataLoader(train_set, batch_size=hyperparameter.batch_size, shuffle=True, num_workers=hyperparameter.num_workers)
    validation_loader = DataLoader(validation_set, batch_size=hyperparameter.batch_size, shuffle=False, num_workers=hyperparameter.num_workers)
    model = torchvision.models.resnet18(pretrained=True)
    num_channels = len(Channels)
    print('The number of channels is :',num_channels)
    model.conv1 = nn.Conv2d(num_channels, 64, (7, 2), padding=3, bias=False)
    model.fc = nn.Linear(in_features=512, out_features=1)
    model.to(device)
    visualize_prediction(model,dataset=train_loader)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameter.learning_rate)
    return model, train_loader, validation_loader, criterion, optimizer

def train(model, train_loader, criterion, optimizer, hyperparameter,second_loss):
    example_ct = 0
    batch_ct = 0
    loss_list =[]
    loss2_list =[]
    for batch_idx, (data, targets) in enumerate(train_loader):
        loss, loss2 = train_batch(data, targets, model, optimizer, criterion,second_loss)
        example_ct += len(data)
        batch_ct += 1
        loss_list.append(loss.item())
        loss2_list.append(loss2.item())
        if((batch_ct + 1 ) % 5) == 0:
            train_log(loss, example_ct, loss2,hyperparameter.num_epochs)
    wandb.log({'train_average_loss': np.mean(loss_list)})
    wandb.log({'train_average_l1_loss': np.mean(loss2_list)})
    #break

def train_batch(data, targets, model, optimizer, criterion,second_loss):
    data, targets = data.to(device) , targets.to(device)
    scores = model(data)
    loss = criterion(scores, targets)
    loss2 = second_loss(scores,targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, loss2

def train_log(loss, example_ct, loss2,epoch):
    train_loss = float(loss)
    train_loss2 = float(loss2)
    wandb.log({'train_per_sample_loss': train_loss})
    wandb.log({'train_per_sample_l1_loss': train_loss2})
    print(f"loss after"+str(example_ct).zfill(5)+f"examples:{train_loss:.3f}")

def validation(model, validation_loader, criterion):
    valid_loss = []
    model.eval()
    with torch.no_grad():
        for data, targets in validation_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            valid_loss.append(loss.item())
        wandb.log({'Validation_average_loss': np.mean(valid_loss)})

def visualize_prediction(model,dataset,num_images=10):
      model.eval()
      with torch.no_grad():
         for i,(data,targets) in enumerate(dataset):
             data = data.to(device)
             targets = targets.to(device)
             outputs = model(data)
             image = data[i]
             plt.imshow(image.cpu().squeeze(), cmap='gray')
             ground_truth = str(targets[i])
             model_predicted = str(outputs[i])
             title=(f'known NC:{ground_truth[7:15]}\u03BCm predicted NC:{model_predicted[8:14]}\u03BCm')
             plt.title(title)
             plt.show()
             plt.savefig('/home/zachary/Desktop/Research/Deep learning/CheckPoints/Resnet18/CAKI2/All_Channels/test/'+ str(i)) # show the image
             if i  == num_images:
                 return



# Saving the model
save_path = '/home/zachary/Desktop/Research/Deep learning/CheckPoints/Resnet18/CAKI2/All_Channels/test/model.pth'
torch.save(model_pipeline(hyperparameter).state_dict(), save_path)
# mlp = model_pipeline(config)
# mlp.load_state_dict(torch.load(save_path))
# mlp.eval()

