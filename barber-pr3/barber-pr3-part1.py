# Ryan Barber
# CISC 442
# PR 3 Deep learning


#import torchvision
#train_data=torchvision.datasets.CIFAR100(root='dataset',download=True)

# Install pytorch library first from documentation online
# You can also use Google Colab (uses Jupyter Notebook) for this assignment
import os
import torch
import torchvision
import torch.utils.data as data_utils
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

############################### CUDA SETTINGS ##################################
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

###############################HYPER-PARAMETERS ################################
# Change these as asked in assignment
n_epochs = 30 # Number of epochs
batch_size_train = 16 # Batch size for training
batch_size_test = 256 # Batch size for testing

###############################SETUP DATA ######################################
transform = transforms.Compose(
    [torchvision.transforms.Resize(32),
     transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5,), (0.5,))]
)

# Change dataset to CIFAR here
# dataset is directory where CIFAR is stored
# You would need to download the dataset first. Please look at PyTorch
# documentation for that
# I HAVE CHANGED MNIST TO CIFAR - Ryan
train_data = datasets.CIFAR100('dataset', train=True, transform=transform)
test_data = datasets.CIFAR100('dataset', train=False, transform=transform)

# Note: DON't subset for assignment, i.e, comment or remove next 4 lines of code
#train_idx = torch.arange(10000)
#test_idx = torch.arange(500)
#train_data = data_utils.Subset(train_data, train_idx)
#test_data = data_utils.Subset(train_data, test_idx)

print('Train size: {}'.format(len(train_data)))
print('Test size: {}'.format(len(test_data)))

# Data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test, shuffle=True)

#Pro-tip: Try to visualize if images and labels are correctly loaded
# before training your network. Use matplotlib. Write your own code here.
#import matplotlib.pyplot as plt
#import numpy as np
#def imshow(img):
#    img = img/2+0.5
#    npimg = img.numpy()
#    plt.imshow(np.transpose(npimg, (1,2,0)))
#    plt.show()
#dataiter = iter(train_loader)
#images, labels = dataiter.next()
#imshow(torchvision.utils.make_grid(images))


###############################SETUP MODEL #####################################
class Net(nn.Module):
    def __init__(self):
        
        #super().__init__()
        #self.conv1 = nn.Conv2d(1, 6, 5)
        #self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 10)
        
        super().__init__()
        self.conv1 = nn.Conv2d(3, 30, 5)
        self.conv2 = nn.Conv2d(30, 60, 3)
        self.conv3 = nn.Conv2d(60, 120, 3)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(120 * 4 * 4, 240)
        self.fc2 = nn.Linear(240, 100)

        

    def forward(self, x):
        #print(x.shape) #32
        x = F.relu(self.conv1(x))
        #print(x.shape) #28
        x = self.pool(x)
        #print(x.shape) #14
        x = F.relu(self.conv2(x))
        #print(x.shape) #12
        x = self.pool(x)
        #print(x.shape) #6
        x = F.relu(self.conv3(x))
        #print(x.shape) #4
        #x = self.pool(x)
        #print(x.shape) #2
        x = x.view(-1, 120 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
print("GPU:" ,torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
# Loss function
criterion = nn.CrossEntropyLoss()
# Optimizer
# Learning Rate = 0.001
# Momentum = 0.9
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#################################### TRAIN #####################################

# I am making this a function

def train_cifar():
	# loop over the dataset multiple times
	for epoch in range(n_epochs):
	    print('Epoch: {0}'.format(epoch))
	    running_loss = 0.0
	    # loop over mini-batch
	    for i, data in enumerate(train_loader, 0):
	        # get the inputs; data is a list of [inputs, labels]
	        inputs, labels = data[0].to(device), data[1].to(device)
	        # zero the parameter gradients
	        optimizer.zero_grad()
	        # forward + backward + optimize
	        outputs = net(inputs)
	        loss = criterion(outputs, labels)
	        loss.backward()
	        optimizer.step()
	        # print statistics
	        running_loss += loss.item()
	print('Finished Training')
	# change it to something like cifar.pth - CHANGED FROM mnist_net.pth
	PATH = './cifar_net.pth'
	#PATH = './cifar_net_cpu.pth'	# specify CPU
	torch.save(net.state_dict(), PATH)
#========================================================

################ CALL TRAINING FUNCTION ########################################

# Call function to train
# uncomment to train network
#train_cifar()

##################################### TEST #####################################
# Reinitialize model and load weights
PATH = './cifar_net.pth'	# created with GPU
#PATH = './cifar_net_cpu.pth'	# created with CPU
net = Net().to(device)
net.load_state_dict(torch.load(PATH))

# test on training images
correct = 0
total = 0
with torch.no_grad():
    for data in train_loader: # USING train_loader data
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Predictions on TRAINING SET')
print('Correct Predictions: {0}'.format(correct))
print('Total Predictions: {0}'.format(total))
print('Accuracy of the network on the TRIANING SET images: %d %%' % (100 * correct / total))

# test on training images
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader: # USING test_loader 
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Predictions on TEST SET')
print('Correct Predictions: {0}'.format(correct))
print('Total Predictions: {0}'.format(total))
print('Accuracy of the network on the TEST SET images: %d %%' % (100 * correct / total))

