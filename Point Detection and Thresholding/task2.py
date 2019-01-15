#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 23:03:03 2018

@author: venktesh
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from collections import Counter

# READING THE IMAGE TURBINE BLADE
def point_detection():
    img = cv2.imread('./original_imgs/turbine-blade.jpg',0)  
    
    pixel_value = []
    pixel_loc = []
    
    # INITIALIZING THE 5*5 KERNEL. SO while traversing/convolving the image uniform 
    # gray level intensties the kernel will make them s zero because of the weighted sum.
    # Whenever there is a sudden change in gray level intensities the convolution will make it as a 
    # high value.
    Gx = [[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,24,-1,-1],
              [-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1]]
    
    arr1 = np.array(Gx)
    lengthimg = np.size(img)
    lengthker = np.size(arr1)
    
    # START OF CONVOLUTION
    img_h = img.shape[0]
    img_w = img.shape[1]
    
    arr1_h = arr1.shape[0]
    
    arr1_w = arr1.shape[1]
    
    h = arr1_h//2
    w = arr1_w//2
    
    img_convx = np.zeros(img.shape)
    for i in range(h,img_h-h):
        for j in range(w,img_w-w):
            sum=0
            for x in range(arr1_h):
                for y in range(arr1_w):
                    sum = sum + (arr1[x][y] * img[i-h+x][j-w+y])
                    img_convx[i][j] = sum
    
    #Taking abosulte value of convolved to counter negative values
    cv2.imwrite('convolved_point.jpg',img_convx)
    img_convx = np.abs(img_convx)

    
    # Performing Thresholding R >= \T\ and take 90% of the thresholded value
    dst_max = int(np.max(img_convx))
    dst_max = dst_max * 0.90
    # Retaining pixels only greater than thresholding value
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if(img_convx[i][j] <= dst_max):
                img_convx[i][j] = 0
            else:
                pixel_value.append((img_convx[i][j]))
                pixel_loc.append((i,j))
    
    # Finding the location of the point
    pixel_loc = np.array(pixel_loc)
    print("LOCATION OF PIXEL")
    print(pixel_loc)
    cv2.putText(img_convx," (249,445)",(pixel_loc[0][1],pixel_loc[0][0]),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,255))
    cv2.namedWindow('res_task2a',cv2.WINDOW_NORMAL)
    cv2.imshow('res_task2a',img_convx)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def segmentation():
    
    #SEGMENTATION OF FILLETS TASK
    img = cv2.imread('./original_imgs/segment.jpg',0)

    #Flattening the image 
    flattened_img = img.flatten()
    nonzero_img = np.nonzero(flattened_img)

    #Using counter in collections to count the number of pixels 
    d = Counter(flattened_img[nonzero_img])
    lists = sorted(d.items())
    x,y = zip(*lists)

    #Plotting the Histogram
    plt.plot(x,y)
    plt.show()

    temp = img
    # From the above obtained Histogram the right most peak is picked which 
    # represents the top most foreground, later using the peak value the foreground
    # is seperated from the background
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] <205:
                temp[i][j]= 0
            else:
                temp[i][j] = 255
    cv2.imshow('temp',temp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('res_tas2b.jpg',temp)

    
def main_func():
    print(' Performing Point Detection ')
    point_detection()
    print(' Performing Segmentation ')
    segmentation()
main_func()

    
