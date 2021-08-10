""" First we import the libraries that we need """
import torch # pytorch
import torch.nn as nn # Pytorch neural network
import torch.optim as optim # pytorch optimization
import torchvision.transforms as transforms # pytorch transforms
import torchvision # torchvision
from torch.utils.data import DataLoader # dataloader
from CustomDatasetv5 import CellDataset # Custom data loader
import numpy as np # numpy
import matplotlib.pyplot as plt # matplotlib for plotting
import wandb # weights and biases for tracking
wandb.login() # weights and biases login


# chose device used for evaluation of model if cuda device = gpu if cpu device = cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # find the device to be used
print('device used : ', device) # Print the device that is used


"""

define hyperparameters in a dictionary named hyperparameter . The hyperparameters defined are learning rate, batch size, 
number of epochs, and number of workers. 
hyperparameters: parameters that control the learning process of the model 
learning rate: parameter that determines the step size at each iteration while moving towards the minimum of the loss function
batch size: the number of training examples used in one iteration 
number of epochs: the number of passes through the dataset 
number of workers: the number of sub processes to use for data loading 
"""
hyperparameter = dict(  # hyper parameters dictionary
    learning_rate=1e-3, # learning rate
    batch_size=16, # patch size
    num_epochs=1, # number of epochs
    num_workers=1, # number of workers
)
# {dict:4} dict with 4 elements {'learning_rate':0.001,'batch_size':16,'num_epochs':1,'num_workers':1}
""" Variables and paths to data """

"""Paths to files where the data is located  """
csv = '/home/zachary/Desktop/DeepLearning/PreProcessing/CAKI2/CAKI2.csv'   # Path to csv file with labels and file names
Ch1 = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/All/Ch1'   # Path to Ch1 images.
train = '/home/zachary/Desktop/DeepLearning/PreProcessing/CAKI2/train/train.csv' # Path to training csv file
validation = '/home/zachary/Desktop/DeepLearning/PreProcessing/CAKI2/test/test.csv' # path to validation csv file

"""Variables"""
loss_list = [] # Empty list used to store loss during training list
validation_loss_list = [] # Empty list used to store loss during validation
transform = transforms.Compose([ # transforms to be applied to the images
            transforms.ToPILImage(), # Transform the image to a PIL image
            transforms.Resize((256, 256)), # Resize the images so that they are 256px by 256px
            transforms.CenterCrop(224), # Center crop the images
            transforms.ToTensor() # convert the images to tensors
        ]) # List with 4 elements [ToPILimage(),Resize(size=(256,256),CenterCrop(size=(224,224)),ToTensor()]
train_set = CellDataset(csv_file=train, root_dir=Ch1, transform=transform) # Load the training data using our custom data loader.

"""
train_set = {CellDataset:876} 
annotations = {pandas dataframe with size (876,2) }
root_dir = '/home/zachary/Desktop/DeepLearning/Dataset/CAKI2/All/Ch1 directory to the images for the data set 
transform = pass images through transform list above line 47-52
"""
validation_set = CellDataset(csv_file=validation, root_dir=Ch1, transform=transform) # Load the validation data using out custom data loader
"""
validation_set = {CellDataset:220}
Annotations = {pandas dataframe with size (220,2) }
root director = Ch1 directory to images 
transform = pass images through transform list line 47-57 
"""
def model_pipeline(hyperparameters): #function to tell weights and biases what to track

    with wandb.init(project='test', config=hyperparameters): # tell weights and biases to get started
        hyperparameter = wandb.config # access all the hyperparameters through the config dictionary and log them with weights and biases
        model, train_loader, validation_loader, criterion, optimizer=make(hyperparameter)  # make the model, training data, validation data, and optimization and use them to train the model
        wandb.watch(model, criterion, log='all', log_freq=10) # telling weights and biases what to watch
        for epoch in range(hyperparameter.num_epochs):  # for epoch in range of the number of epochs specified above in the hyperparameter dictionary
            train(model, train_loader, criterion, optimizer, hyperparameter) # train the model, function inputs = resnet model, training data, loss function (criterion), optimizer, and hyperparameter dictionary
            validation(model, validation_loader, criterion) # validate the model, function inputs = resent model, validation data, and loss function (criterion)

            """
            For loop that says for each epoch that is in the range of the number of epochs specified in the config dictionary above train the model and then validate the model 
            - model = resnet 18 model pretrained on image net 
            - train_loader = Load the training data using our custom data loader 
            - criterion = loss function 
            - optimizer = ADAM using our hyperparameters defined in the dictionary above 
            -  validation_loader = Load the validation data using our custom data loader
            """
    return model # return the model





def make(hyperparameter): # function to make the data and the model
    train_loader = DataLoader(train_set, batch_size=hyperparameter.batch_size, shuffle=True, num_workers=hyperparameter.num_workers) # Load the training data using the torch.util data loader
    """ Load the training data using torch.util.dataloader where:
    train_set is the training data loaded with our custom data loader, 
    batch_size is the batch size defined in the config dictionary above
    shuffle = True  data reshuffles at every epoch
    num_workers is the number of workers defined in the config dictionary above """
    validation_loader = DataLoader(validation_set, batch_size=hyperparameter.batch_size, shuffle=False, num_workers=hyperparameter.num_workers) # Load the validation data using the torch.util data loader
    """
     Load the validation data using the torch.util.dataloader where:
    validation_set is the validation data loaded using our custom data loader
    batch_size is the batch size set to the batch size defined in the config dictionary above 
    shuffle = False data is not reshuffled at every epoch
    num_workers is the number of workers defined in the config dictionary above  
    """
    # Define our Resnet model
    model = torchvision.models.resnet18(pretrained=True) # define our model as a resnet18 model pretrained = true means the the model we are using is pretrained on the imagenet data set.
    model.conv1 = nn.Conv2d(1, 64, (7, 2), padding=3, bias=False) # Here we had to change the input of the first layer for 3 (rgb) to 1(greyscale) so that we could use our dataset of greyscale cell images
    """model.conv1 inputs = number of channels in, number of channels out, kernel_size, stride, padding, bias
    stride: controls the stride for the cross-correlation, a single number or a one element tuple 
    padding: controls the amount of padding applied to the input. It can be either a string {'valid','same'} or a tuple of ints giving the amount of implicit padding applied on both sides 
    in_channels = number of channels in the input image 
    out_channels = number of channels produced by the convolution
    kernel_size = Size of the convolving kernel 
    stride = Stride of the convolution 
    padding = padding added to both sides of the input. Default 0 
    bias = if True, adds a learnable bias to the output. Default: True"""
    model.fc = nn.Linear(in_features=512, out_features=1) # we alse had the change the fully connected layer so that it takes a (256 + 256 = 512) input and output just 1 feature because our problem is a regression problem
    model.to(device) # send the model to the device defined above either GPU or CPU
    visualize_prediction(model.to(device),validation_loader)

    # Define the loss and the optimizer
    criterion = nn.MSELoss() # criterion is defined as the loss function in this case the loss function is loaded from the torch neural network library and is defined as the mean square error loss function
    optimizer = optim.Adam(model.parameters(), lr=hyperparameter.learning_rate) #  Here the optimizer is defined using the pytoch optimizer library and here we used the Adam optimizer and the learning rate used with the optimizer is the learning rate defined in the config dictionary above
    """Optimizer: parameter used to reduce the loss function"""
    return model, train_loader, validation_loader, criterion, optimizer # This function will return the model, the training and validation data the loss function and the optimizer



def train(model, train_loader, criterion, optimizer, hyperparameter): # function that is used to train the model. This function defines how the model well be trained
    # total_batches = len(train_loader) * config.num_epochs
    example_ct = 0 # example count
    batch_ct = 0 # batch count

    for batch_idx, (data, targets) in enumerate(train_loader): # for images and labels in training loader train the images
        # targets = labels
        loss = train_batch(data, targets, model, optimizer, criterion) # training loss of our model
        example_ct += len(data) # I do not understand what this is doing
        batch_ct += 1 # I also do not understand what this is doing
        loss_list.append(loss.item()) # add our loss defined in line 117 to our  empty loss list defined above in line 45
        if((batch_ct + 1 ) % 5) == 0: # report metrics every 5th batch and log the loss for weights and biases
            train_log(loss, example_ct, hyperparameter.num_epochs)
            #break

def train_batch(data, targets, model, optimizer, criterion): # function that defines the batches for training
    data, targets = data.to(device) , targets.to(device)


    # data = data.to(device) # send the images to the device either GPU or CPU
    # targets = targets.to(device) # Send the labels to the device either GPU or CPU

    # forward pass
    scores = model(data) # scores = the model trained on the images
    loss = criterion(scores, targets) # loss = MSE loss evaluated with the scores and the labels

    # backward pass
    optimizer.zero_grad() # Pass the optimizer through the model
    loss.backward()
    # step with optimizer
    optimizer.step()

    return loss # return the loss
def train_log(loss, example_ct, epoch): # define what we want to log with weights and biases
    train_loss = float(loss) # convert the loss from a torch float tensor to a float
    wandb.log({'train_per_sample_loss': train_loss}) # log the train per sample loss
    wandb.log({'train_average_loss': np.mean(loss_list)}) # calculate and log the average loss
    print(f"loss after"+str(example_ct).zfill(5)+f"examples:{train_loss:.3f}") # Print the loss after each step.

def validation(model, validation_loader, criterion): # Function to define how we want the model to validate

    valid_loss = 0 # validation loss = 0
    model.eval() # Evaluate the model
# Run the model on some test examples
    with torch.no_grad(): # I really do not know what this while loop means
        correct = 0 # I am not to sure what this is for
        total = 0  # I am not to sure what this is for
        for data, targets in validation_loader: # for images and targets in validation loader validate our model
            """
            For loop for validating our model. For images and targets in our validation loader validate our model.
            Similar to the training for loop used in our training function 
            """
            data, targets = data.to(device), targets.to(device)
            # data = data.to(device) # send the images to the device either CPU or GPU
            # targets.to(device) # send the labels to the device either CPU or GPU
            outputs = model(data) # outputs = the model trained on our images
            loss = criterion(outputs, targets) # loss = the loss function evaluated at the model trained on the images and the labels
            valid_loss += loss.item() # Add this loss defined in the line above to the validation loss defined in line 147
            _, predicted = torch.max(outputs.data,1) # returns the index of the highest prediction
            # torch.max returns the maximum value of all elements in the input tensor



        wandb.log({'Validation_average_loss': np.mean(valid_loss)}) # with weights and biases calculate the average validation loss and then log it


def visualize_prediction(model,validation_loader,num_images=10): # Function to visualize predictions made with the model. This function takes an input our model, validation data and the number of images you want to display
      model.eval() # Evaluate the model
      images_so_far = 0 # images that have been shown so far
      with torch.no_grad(): # Using torch
         for i,(data,targets) in enumerate(validation_loader): # for image in data and targets located in the validation loader
             data = data.to(device) # send the images to the device
             targets = targets.to(device) # send the labels to the device
             outputs = model(data) # run the images through the model
             _,predicted = torch.max(outputs,1) # returns the index of the highest prediction
             images_so_far +=1 # add one to the number of images used so far
             image = data[i] # image = image tensor indexed at i
             plt.imshow(image.cpu().squeeze(), cmap='gray') # plot the image
             ground_truth = targets[i]
             model_predicted = _[i]
             title = ('ground truth',ground_truth,'model predicted',model_predicted) # set the title
             plt.title(title) # add the title to the image
             plt.show() # show the image
             if images_so_far == num_images: # keep going over images until num_images = images_so_far
                 return



# Saving the model
save_path = './mlp.pth'
torch.save(model_pipeline(hyperparameter).state_dict(), save_path)
# mlp = model_pipeline(config)
# mlp.load_state_dict(torch.load(save_path))
# mlp.eval()
