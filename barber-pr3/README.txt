Ryan Barber
CISC 442
PR 3

I use code from 'class.py' file provided by Vinit.

Source for Part 2 is mentioned here and in the file barber-pr3-part2.py
	Source: https://pytorch.org/hub/pytorch_vision_fcn_resnet101/

Modules used:
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


Part 1

	Run instructions
		python3 barber-pr3-part1.py

	I have included the trained network for both GPU and CPU
		GPU version: cifar_net.pth
		CPU version: cifar_net_cpu.pth

	Currently the code is setup to test using the GPU version
	
	If you don't have GPU:
		See lines 164, 165 in barber-pr3-part1.py and switch the commented lines
	
	I have changed the code to have the training part in a function.
	If you wish to train the network again, uncomment the line of code
	under the CALL TRAINING FUNCTION section (line 160). Recommend having GPU, otherwise
	the training will take a very long time.

	Otherwise, running the code will just run test with the network.

	Output:
		First shows the accuracy of the network on the TRAINING images.
		Second shows the accuracy of the network on the TEST images.

	Submission:
		a) How to run code
			python3 barber_pr3_part1.py

			You can use any python command that has the required modules

		b) Kernel and Stride size of convolution layers
			Convolution layer 1 - "conv1"
				Kernel = 5x5
				Stride = 1x1
			Convolution layer 2 - "conv2"
				Kernel = 3x3
				Stride = 1x1
			Convolution layer 3 - "conv3"
				Kernel = 3x3
				Stride = 1x1

		c) Kernel size of pooling layer
			Pooling layer - "pool"
				Kernel = 2x2

		d) Learning Rate, Momentum, Batch size
			Learning Rate = 0.001
			Momentum = 0.9
			Batch size (training set) = 16
			Batch size (test set) = 256

		e) Accuracy
			Accuracy on test set = 33%

			Accuracy on training set = 93%


	Summary:
		I was able to get the accuracy on the training set to 93%,
		however, the accuracy on the test set was very low being only
		33%.

		I experimented with the network to try to get the accuracy on the test
		set higher but I could only get it to about 38-39%
 
		
Part 2

	Run Instructions
		python3 barber-pr3-part2.py

	The code is currently set up to run on image 1 ("img1.jpg").
	It will display the grid of feature maps and the final segmented
	image when you run it. It will not write them out.
	
	Output
		See each folder "imgXresults" where X = image number
		Each folder has the original image, the grid of feature maps and
		the final segmentation image.

	Feature Maps:
		The feature maps were tricky to look at. They had negative values in
		them and the values are all quite low for the range 0 - 255. When I
		displayed the feature maps with matplotlib or openCV, the images were
		very dark. To make them show up more, I multiplied the feature maps by
		10. This makes the features more visible.
		
		The final feature maps were fairly accurate. I got my input images from
		Pascal VOC dataset and I tried a bunch of images until the final image
		was the best. Sometimes the image was not detecting the objects if they
		were too small or it was dectecting something that wasn't there becuase
		of a complex texture. However, the images I used had the best results I
		could find. The results were good at showing the general shape of the
		objects but didn't always get every part of the object. Also, in my img4
		result, the cat is partialy classified as something else (probably dog).
		
		I did not change the input size of the images so I removed that from the
		transform funciton. If I made the images smaller, the feature maps would
		not find the features. The images from the dataset I used were all roughly
		the same size (~375x500) which I thought was ideal size. I tried to use
		large images like 1920x1080 and larger but they had lots of noisy
		misclassification spots. 

