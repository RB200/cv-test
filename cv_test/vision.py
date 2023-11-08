import cv2 as cv
import numpy as np
import os
#read the image
#len_files = len([name for name in os.listdir('./pics/reference/') if os.path.isfile(name)])
count = 0
dir_path = "./pics/reference"
for path in os.listdir(dir_path):
    if os.path.isfile(os.path.join(dir_path,path)):
        count += 1

for i in range(count):

    image = cv.imread(f"./pics/reference/IMG_23{38+i}.jpg")
    print(f"Image {i} processing")
    #convert the image to RGB (images are read in BGR in OpenCV)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    #split the image into its three channels
    (R,G,B) = cv.split(image)


    R = cv.subtract(R,G)
    R = cv.subtract(R,B)

    #create named windows for each of the images we are going to display

    #cv.namedWindow("Red - Green", cv.WINDOW_AUTOSIZE)

    #display the images
    R2 = cv.resize(R,(1008,567))

    #cv.imshow("Red - Green", R2)
    cv.imwrite(f"./pics/results/RedMinusGreenAndBlue{i}.jpg",R2)

    ################
    image = cv.imread(f"./pics/results/RedMinusGreenAndBlue{i}.jpg")
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) 
    
    # Find Canny edges 
    edged = cv.Canny(gray, 30, 200) 
    contours, hierarchy = cv.findContours(edged,  
        cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 
    
    new_contours = []
    for c in contours:
        if len(c) > 150:
            approx = cv.approxPolyDP(c, 0.03*cv.arcLength(c, True), True)  # form approximations of the shapes ...
            if len(approx) == 3:
                new_contours.append(c)
    
    cv.drawContours(image, new_contours, -1, (0, 255, 0), 3) 

    #cv.imshow('Canny Edges After Contouring', edged) 

    #cv.imshow("Cool outlines",image)
    cv.waitKey(0) 

    cv.imwrite(f"./pics/results/RMGABOutlines{i}.jpg",image)
    if cv.waitKey(0):
        cv.destroyAllWindows()