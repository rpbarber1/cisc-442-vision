Ryan Barber
CISC 442
PR1

Run Commands
    $ python3 barber-pr1-partA.py

    $ python3 barber-pr1-partB.py

The programs run all at once. Once you close a matplotlib window,
another will pop up immediately with the next thing.

Part A

    This part runs very slow. It may take several minutes for the example.

    First Test:
    	The first thing that happens is I do a test of Reconstruct().
    	This demonstrates that all the previous functions are working.
    	This will run by itself, just wait.
    	Once you close the output window, the next test starts.
    	
    Second Test/Example:
    	For Blending, I have left 1 test un-commented so you can do it yourself.
    	This is the affine unwarping and blending test.
    		First, a window will pop up where you pick 3 points on the left
    		image and then 3 corresponding points on the right image (in the
    		same order). These instructions are in the plot title. 
    		
    		Second, the original image and the unwarped image will appear.
    			The unwarped image will be small, so I recommend
    			enlarging the window for this part
    			
    		Pick one point on the right (unwarped) image that corresponds to
    		the right edge of the left image.
    			This is like drawing the bondary except you just click
    			a point that would be on boundary line.
    	

	Results/Submitted Images
		The affine and perspective unwarped images are correctly blended,
		but due to human error in clicking the matching points, the homograpy
		is not perfect (it is close). 
		

	General Comments:
    		I have commented out the test for each individual function except
    		Reconstruct() because it uses all of them. 
    		
    		For blending, I commented out 3 of the 4 parts. I only left affine
    		unwarping + blending becuase it is faster and demonstrates unwarping.



Part B

    The functions are in the helpers_partB.py file.
    
    THIS PART IS ALL INTERACTIVE.

    The title of the matplotlib window has the basic instruction
    Instructions:
    
        PERSPECTIVE

        First window - select 4 points from original image.
                        select 4 opints from warped image (in same order)
        Second window - select 7 points from original image.
                        select 7 opints from warped image (in same order)

        Now a window will pop up with all the Perspective images.
        Once you close the window, the error will be printed to terminal.

        AFFINE

        Third window  - select 3 points from the original image.
                        select 3 points from warped image (in same order)
        Fourth window  - select 4 points from the original image.
                        select 4 points from warped image (in same order)

        Now a window will pop up with all the Affine images.
        Once you close the window, the error will be printed to terminal
        
    Results:
    	I have submitted results of this section with screenshots and text file
    	with the homography matrixes and error numbers

    General Notes
    	The homography depends on the accuracy of the clicks. 
	I have set the input to wait forever so there is no rush to pick points.
	
	
	If there is a very strange warping, for example with Perspective + Over Constrained,
	you most likley clicked the matching points out of order or double clicked a point.
	This was the most common issue I was having. 
	
	I would ctrl+C and restart the program if you make a mistake clicking points.
	Try to get accurate points by making window bigger or take your time chosing points
    	
    	
    	
    	
