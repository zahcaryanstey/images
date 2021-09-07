import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from DataLoader_All_Channels import CellDataset
from CIFAR_10_NN import Net
import numpy as np
import matplotlib.pyplot as plt
import wandb
wandb.login()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device being used :',device)

Data_set = 'SKBR3'
File_name = 'Ch1'
hyperparameter = dict(
    learning_rate = 1e-2,
    batch_size = 16,
    num_epochs = 10,
    num_workers = 2,
    model_name = 'Resnet34',


)
"""
Add Paths to hyperparameter dictionary above 
"""
csv = '/home/zachary/Desktop/DeepLearning/PreProcessing/'+ Data_set +'/' + Data_set+'.csv'
Ch1 = '/home/zachary/Desktop/DeepLearning/Dataset/'+Data_set + '/All/Ch1'
Ch7 = '/home/zachary/Desktop/DeepLearning/Dataset/' + Data_set + '/All/Ch7'
Ch11 = '/home/zachary/Desktop/DeepLearning/Dataset/' + Data_set + '/All/Ch11'
train_path = '/home/zachary/Desktop/DeepLearning/PreProcessing/' + Data_set +'/'+Data_set+'train.csv'
validation_path = '/home/zachary/Desktop/DeepLearning/PreProcessing/' + Data_set +'/'+Data_set+'test.csv'

Channels = [Ch1]


transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            # transforms.Resize((32, 32)),
            # transforms.CenterCrop(224),
            transforms.ToTensor()
        ])


train_set = CellDataset(csv_file=train_path, root_dir=Channels, transform=transform)
validation_set = CellDataset(csv_file=validation_path, root_dir=Channels, transform=transform)


def model_pipeline(hyperparameters):
    with wandb.init(project='HT29-All_Channels-Resnet34-Lr1e-2-SGD',config=hyperparameters):
        hyperparameter = wandb.config
        model, train_loader, validation_loader,criterion,optimezer=make(hyperparameter)
        second_loss = nn.L1Loss(reduction='mean')
        wandb.watch(model,criterion,log='all',log_freq=10)
        for epoch in range(hyperparameter.num_epochs):
            train(model,train_loader,criterion,optimezer,hyperparameter,second_loss)
            validation(model,validation_loader,criterion)

    return model

def make(hyperparameter): # function to make the data and the model
    train_loader = DataLoader(train_set, batch_size=hyperparameter.batch_size, shuffle=True, num_workers=hyperparameter.num_workers)
    validation_loader = DataLoader(validation_set, batch_size=hyperparameter.batch_size, shuffle=False, num_workers=hyperparameter.num_workers)
    num_channels = len(Channels)
    if hyperparameter.model_name == 'CIFAR':
        model = Net(num_channels)
    elif hyperparameter.model_name =='Resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(num_channels, 64, (7, 2), padding=0, bias=False)
        model.fc = nn.Linear(in_features=512, out_features=1)
    elif hyperparameter.model_name == 'Resnet34':
        model = torchvision.models.resnet34(pretrained=True)
        model.conv1 = nn.Conv2d(num_channels, 64, (7, 2), padding=0, bias=False)
        model.fc = nn.Linear(in_features=512, out_features=1)
    elif hyperparameter.model_name == 'Resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.conv1 = nn.Conv2d(num_channels, 64, (7, 2), padding=0, bias=False)
        model.fc = nn.Linear(in_features=512, out_features=1)

    print('The number of channels is :',num_channels)
    model.to(device)
    visualize_prediction(model,dataset=train_loader)
    # criterion = nn.L1Loss(reduction='mean')
    criterion = nn.MSELoss(reduction='mean')
    # optimizer = optim.Adam(model.parameters(), lr=hyperparameter.learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=hyperparameter.learning_rate)
    return model, train_loader, validation_loader, criterion, optimizer

def train(model, train_loader, criterion, optimizer, hyperparameter,second_loss):
    example_ct = 0
    batch_ct = 0
    loss_list =[]
    loss2_list =[]
    model.train()
    # for batch_idx, (data, targets) in enumerate(train_loader):
    for batch_idx, (data1, targets1) in enumerate(train_loader):
        if batch_idx == 0:
            data = data1
            # data = torch.zeros(data1.shape).cuda()
            targets = targets1
        loss, loss2 = train_batch(data, targets, model, optimizer, criterion,second_loss)
        example_ct += len(data)
        batch_ct += 1
        loss_list.append(loss.item())
        loss2_list.append(loss2.item())
        # if((batch_ct + 1 ) % 5) == 0:
        train_log(loss, example_ct, loss2,hyperparameter.num_epochs)
    wandb.log({'train_average_loss': np.mean(loss_list)})
    wandb.log({'train_average_l1_loss': np.mean(loss2_list)})
    #break

def train_batch(data, targets, model, optimizer, criterion,second_loss):
    data, targets = data.to(device) , targets.to(device).unsqueeze(1)
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
    print(f"loss after "+str(example_ct).zfill(5)+f" examples:{train_loss:.6f}")

def validation(model, validation_loader, criterion):
    valid_loss = []
    model.eval()
    with torch.no_grad():
        for data, targets in validation_loader:
            data, targets = data.to(device), targets.to(device).unsqueeze(1)
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
             if len(Channels) == 1:
                 plt.imshow(image.cpu().squeeze(),cmap='gray')
                 ground_truth = str(targets[i])
                 model_predicted = str(outputs[i])
                 title = (f'known NC:{ground_truth[7:15]}\u03BCm predicted NC:{model_predicted[8:14]}\u03BCm')
                 plt.title(title)
                 plt.savefig('/home/zachary/Desktop/Research/Deep learning/CheckPoints/Resnet34/' + Data_set + '/' + File_name + '/' + str(i))  # show the image
                 if i  == num_images: # keep going over images until num_images = images_so_far
                     return
             elif len(Channels) == 3:
                 for data in Channels:
                    f,axarr = plt.subplots(1,3)
                    axarr[0].imshow(image[0].cpu(),cmap='gray')
                    axarr[1].imshow(image[1].cpu(),cmap='gray')
                    axarr[2].imshow(image[2].cpu(),cmap='gray')

                    # plt.imshow(image[2].cpu(),cmap='gray')
                 # plt.imshow(image.permute(1,2,0).cpu(), cmap='gray')
                    ground_truth = str(targets[i])
                    model_predicted = str(outputs[i])
                    title=(f'known NC:{ground_truth[7:15]}\u03BCm predicted NC:{model_predicted[8:14]}\u03BCm')

                    plt.suptitle(title)
                    # plt.show()
                    f.savefig('/home/zachary/Desktop/Research/Deep learning/CheckPoints/Resnet34/' + Data_set +'/' + File_name + '/'+ str(i)) # show the image
                    if i  == num_images:
                        return




# Saving the model
save_path = '/home/zachary/Desktop/Research/Deep learning/CheckPoints/Resnet34/' + Data_set +'/' + File_name +'/model.pth'
torch.save(model_pipeline(hyperparameter).state_dict(), save_path)
# mlp = model_pipeline(config)
# mlp.load_state_dict(torch.load(save_path))
# mlp.eval()

