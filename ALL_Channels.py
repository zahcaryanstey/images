# Import depndinces
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
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import wandb
wandb.login()

# Set device and then print out which device is being used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device being used :',device)

# Set the dataset to be used and the file name to save results to.
Data_set = 'HT29' # Enter the name of the data set to be used HT29, SKBR3, CAKI2
File_name = 'test' # Enter the file name that you want information to be saved to as well as the channels to be used: Ch1, Ch7, Ch11, test


# Create a dictionary called hyperparameter to hold some useful information for our model:
        # Learning rate
        # batch size
        # number of epochs
        # number of workers
        # model name
hyperparameter = dict(
    learning_rate = 2e-2,
    batch_size = 16,
    num_epochs = 10,
    num_workers = 2,
    model_name = 'Resnet18',


)

# Set the paths to the data. using the Data_set variable from above to select the correct paths.
csv = '/home/zachary/Desktop/Data_set/Csv_files/'+Data_set + '/' + Data_set +'.csv'
Ch1 = '/home/zachary/Desktop/DeepLearning/Dataset/'+Data_set + '/All/Ch1'
Ch7 = '/home/zachary/Desktop/DeepLearning/Dataset/' + Data_set + '/All/Ch7'
Ch11 = '/home/zachary/Desktop/DeepLearning/Dataset/' + Data_set + '/All/Ch11'
train_path = '/home/zachary/Desktop/Data_set/Csv_files/' +Data_set +'/' + Data_set +'train.csv'
validation_path = '/home/zachary/Desktop/Data_set/Csv_files/' + Data_set + '/' + Data_set +'test.csv'


# Using the file name above set the channel list to be used for training and validation.
if File_name == 'Ch1':
    Channels = [Ch1]
elif File_name == 'Ch7':
    Channels = [Ch7]
elif File_name == 'Ch11':
    Channels = [Ch11]
elif File_name == 'All_Channels':
    Channels = [Ch1,Ch7,Ch11]
elif File_name == 'test':
    Channels = [Ch1]

# Set the transform operation to be used on the images:
    # Convert to pil image
    # Re size
    # center crop
    # Convert to tensor
transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            # transforms.Resize((32, 32)),
            # transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

# Create the training set
train_set = CellDataset(csv_file=train_path, root_dir= Channels, transform=transform)

# Plot a histogram of the training data count vs Nc ratio
fig = plt.figure()
sns.histplot(train_set.annotations.NC,color='b')
plt.xlabel(' NC Ratio [\u03BCm]/[\u03BCm]')
plt.ylabel('Count')
plt.title('Training Set')
plt.savefig('Training_Historgram.png')


# Create the validation set
validation_set = CellDataset(csv_file=validation_path, root_dir=Channels, transform=transform)

# Plot a histogram of the validation set
fig2 = plt.figure()
sns.histplot(validation_set.annotations.NC,color='b')
plt.xlabel(' NC Ratio [\u03BCm]/[\u03BCm]')
plt.ylabel('Count')
plt.title('Validation Set')
plt.savefig('Validation_Histogram.png')

# Define the model pipeline
def model_pipeline(hyperparameters):
    with wandb.init(project='NC Ratio ',config=hyperparameters): # Using weights and biases use the project NC ratio as well as the hyperparameter dictionary
        hyperparameter = wandb.config
        model, train_loader, validation_loader,criterion,optimezer=make(hyperparameter) # Use model, training data, Validation data, loss and optimizer
        second_loss = nn.L1Loss(reduction='mean') # L1 loss
        wandb.watch(model,criterion,log='all',log_freq=10)
        for epoch in range(hyperparameter.num_epochs): # For each epoch train and validate the model
            train(model,train_loader,criterion,optimezer,hyperparameter,second_loss) # call on the training function using model, training data, loss, L1 loss, optimizer and hyperparameter dict
            validation(model,validation_loader,criterion) # Call on the validation function using model, validation data, and loss

    return model

# Use the hyperparameters to make the model
def make(hyperparameter): # function to make the data and the model
    train_loader = DataLoader(train_set, batch_size=hyperparameter.batch_size, shuffle=True, num_workers=hyperparameter.num_workers) # Load the training data
    validation_loader = DataLoader(validation_set, batch_size=hyperparameter.batch_size, shuffle=False, num_workers=hyperparameter.num_workers) # Load the Validation data
    num_channels = len(Channels)
    print('The number of channels being used is: ',num_channels)

    # finding the model using the model stated in hyperparameter dict
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
        model.fc = nn.Linear(in_features=2048, out_features=1)
    model.to(device) # Send model to device
    visualize_prediction(model,dataset=train_loader) # Call on the visualize function.
    # criterion = nn.L1Loss(reduction='mean')
    criterion = nn.MSELoss(reduction='mean')
    # optimizer = optim.Adam(model.parameters(), lr=hyperparameter.learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=hyperparameter.learning_rate)
    return model, train_loader, validation_loader, criterion, optimizer


# Define how to train the model
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
            targets = targets1.unsqueeze(1)
        loss, loss2 = train_batch(data, targets, model, optimizer, criterion,second_loss)
        example_ct += len(data)
        batch_ct += 1
        loss_list.append(loss.item())
        loss2_list.append(loss2.item())
        # if((batch_ct + 1 ) % 5) == 0:
        train_log(loss, example_ct, loss2,hyperparameter.num_epochs)
        break
    wandb.log({'train_average_loss': np.mean(loss_list)})
    wandb.log({'train_average_l1_loss': np.mean(loss2_list)})
    #break

# Define the training batch
def train_batch(data, targets, model, optimizer, criterion,second_loss):
    data, targets = data.to(device) , targets.to(device).unsqueeze(1)
    scores = model(data)
    loss = criterion(scores, targets)
    loss2 = second_loss(scores,targets)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    return loss, loss2

# Define what to log with weights and biases
def train_log(loss, example_ct, loss2,epoch):
    train_loss = float(loss)
    train_loss2 = float(loss2)
    wandb.log({'train_per_sample_loss': train_loss})
    wandb.log({'train_per_sample_l1_loss': train_loss2})
    print(f"loss after "+str(example_ct).zfill(5)+f" examples:{train_loss:.6f}")

# Define the validation process for the model
def validation(model, validation_loader, criterion):
    valid_loss = []
    model.eval()
    prediction = []
    with torch.no_grad():
        for data, targets in validation_loader:
            data, targets = data.to(device), targets.to(device).unsqueeze(1)
            outputs = model(data)
            loss = criterion(outputs, targets)
            valid_loss.append(loss.item())

            for image in range(0,data.shape[0]):
                prediction.append(outputs[image].detach().cpu())
                print(prediction)
            # print('model prediction',prediction)

            # predictions = outputs.cpu().numpy()
            # prediction.append(predictions)
            # fig3 = plt.figure(figsize=(4, 2))
            # sns.histplot( predictions,color='b')
            # plt.xlabel('[\u02Bcm]/[\u02Bcm]')
            # plt.ylabel('Count')
            # plt.title('Validation Set')
            # plt.show()
            # print('predictions',predictions)
        # loss = criterion(outputs, targets)
        # valid_loss.append(loss.item())
        # fig3 = plt.figure()
        # sns.histplot(prediction, color ='b')
        # plt.xlabel('[\u02BCm]/[\u02BCm]')
        # plt.ylabel('Count')
        # plt.title('Validation Set')
        # plt.show()
        wandb.log({'Validation_average_loss': np.mean(valid_loss)})

    # Plot histogram here

# Define how we want to visualize our predictions
def visualize_prediction(model,dataset,num_images=1):
      model.eval()
      with torch.no_grad():
         for i,(data,targets) in enumerate(dataset):
             data = data.to(device)
             targets = targets.to(device).unsqueeze(1)
             outputs = model(data)
             for batch_index in range(0,data.shape[0]):
                 image = data[batch_index]
             # image = data[i]
                 if len(Channels) == 1:
                     plt.imshow(image.cpu().squeeze(),cmap='gray')
                     ground_truth = str(targets[batch_index])
                     model_predicted = str(outputs[batch_index])
                     # print(model_predicted)
                     title = (f'known NC:{ground_truth[7:15]}\u03BCm predicted NC:{model_predicted[8:14]}\u03BCm')
                     plt.title(title)
                     plt.savefig('/home/zachary/Desktop/Research/Deep learning/CheckPoints/Resnet18/' + Data_set  +'/' +File_name + '/' + str(i))  # show the image
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
                        ground_truth = str(targets[batch_index])
                        model_predicted = str(outputs[batch_index])
                        title=(f'known NC:{ground_truth[7:15]}\u03BCm predicted NC:{model_predicted[8:14]}\u03BCm')

                        plt.suptitle(title)
                        # plt.show()
                        f.savefig('/home/zachary/Desktop/Research/Deep learning/CheckPoints/Resnet18/' + Data_set +'/'+ File_name + '/'+ str(i)) # show the image
                        if i  == num_images:
                            return




# Saving the model
save_path = '/home/zachary/Desktop/Research/Deep learning/CheckPoints/Resnet18/' + Data_set +'/'+File_name +'/model.pth'
torch.save(model_pipeline(hyperparameter).state_dict(), save_path)
# mlp = model_pipeline(config)
# mlp.load_state_dict(torch.load(save_path))
# mlp.eval()




