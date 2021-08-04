# Ryan Barber
# CISC 442
# PR 3 Deep learning

# Web Source:   https://pytorch.org/hub/pytorch_vision_fcn_resnet101/

# Install pytorch library first from documentation online
# You can also use Google Colab (uses Jupyter Notebook) for this assignment
import os
import torch
import torchvision
import torch.utils.data as data_utils
from torchvision import datasets, models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

############################### CUDA SETTINGS ##################################
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###############################SETUP DATA ######################################
transform = transforms.Compose(
    [transforms.ToTensor(),
     torchvision.transforms.Normalize((0.5,), (0.5,))]
)


############################### GET MODEL ######################################
model = models.segmentation.fcn_resnet50(pretrained = True)
model.eval()
model.to(device)

############################### SET UP INPUT ###############################


# I just change the name for each image and run again
# leave on img1.jpg for submission
input_image = Image.open('img1.jpg')


input_tensor = transform(input_image)
input_batch = input_tensor.unsqueeze(0)

############################### GET OUTPUT #####################################
output = None
with torch.no_grad():
    output = model(input_batch)['out'][0]

#output = list of feature map images
#output_predictions = final color image

output_predictions = output.argmax(0)
print(output_predictions.shape)
print(output.shape)
	
	
################################## TILE FEATURE MAPS ###########################
# Use openCV to concat together
row1 = output[0].numpy()
row2 = output[7].numpy()
row3 = output[14].numpy()
for i in range(1,7):
	row1 = cv2.hconcat([row1, output[i].numpy()])
	row2 = cv2.hconcat([row2, output[i+7].numpy()])
	row3 = cv2.hconcat([row3, output[i+14].numpy()])
grid = cv2.vconcat([row1, row2])
grid = cv2.vconcat([grid, row3])

# See README.txt for more explination
# this is done to show the image better
# comment out to see original
grid = grid* 10

#write out
#cv2.imwrite("img1-feature-maps.jpg", grid)

grid = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
plt.imshow(grid.astype("int64"))
plt.show()

# GET COLOR IMAGE FOR CLASSES
# create a color pallette, selecting a color for each class
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# plot the semantic segmentation predictions of 21 classes in each color
r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)

# write
#plt.imsave("img1-final.jpg", r)

# show
plt.imshow(r)
plt.show()










##########3
