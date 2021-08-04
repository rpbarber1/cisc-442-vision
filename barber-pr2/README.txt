Ryan Barber
CISC 442
PR 2

Images are from the Middlebury datasets. 

Each output has left to right and right to left disparity images,
the validity check image, and the disparity map of harris corner
detection image. I do not include the harris corner detection
images, only the resulting disparity map image. 

I use the same harris corner detection code and parameters as used
in Rohit demo code provided. 

Output 1
	Window size = 9 x 9
	Max Disparity = 20 pixels
	Measure = SSD

Output 2
	Window size = 15 x 15
	Max Disparity = 30 pixels
	Meauser = SAD

Output 3 
	Window size = 9 x 9
	Max Disparity = 20 pixels
	Measure = NCC
	
Output 4
	Window size = 11 x 11
	Max Disparity = 25 pixels
	Measure = SSD
	
Output 5
	Window size = 11 x 11
	Max Disparity = 25 pixels
	Measure = SAD


Running code:
	$ python3 barber-pr2.py 


If you run the code, it will create disparity maps for left-to-right and
right-to-left. Then it creates validity check image, then harris disparity
map image.You can change the parameters in the code under the '#Run' section
at the bottom. The parameters are clearly commented. 

I have commented out the part that gets the output images. You can uncomment this
if you want, it will write the images out that are created.
