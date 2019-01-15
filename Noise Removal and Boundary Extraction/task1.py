import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./original_imgs/noise.jpg',0)

img_h = img.shape[0]
img_w = img.shape[1]
np.shape(img)

kernel = [[1,1,1],[1,1,1],[1,1,1]]
kernel
kernel_h,kernel_w = np.shape(kernel)

kernel_cx = kernel_w // 2
kernel_cy = kernel_h // 2


def erosion(p):
    final_image_erosion = np.zeros(p.shape)
    img = p
    for i in range(img_h):
        for j in range(img_w):
            min = 255
            for x in range(i-kernel_cx,i+kernel_cy+1):
                for y in range(j-kernel_cy,j+kernel_cy+1):
                    if(x>=0 and x< img_h and y>=0 and y<img_w): 
                        if(img[x][y]<min):
                            min = img[x][y]
            final_image_erosion[i][j] = min
    return final_image_erosion

    
def dilation(q):
    final_image_dilation = np.zeros(q.shape)
    img = q
    for i in range(img_h):
        for j in range(img_w):
            max = 0
            for x in range(i-kernel_cx,i+kernel_cx+1):
                for y in range(j-kernel_cy,j+kernel_cy+1):
                    if(x>=0 and x<img_h and y>=0 and y<img_w):
                            if(img[x][y]>max):
                                max = img[x][y]
            final_image_dilation[i][j] = max
    return final_image_dilation

erosion(img)
dilation(img)

def opening_closing_img():
    eroded = erosion(img)
    dilated = dilation(eroded)
    dilated_1 = dilation(dilated)
    opening_out = erosion(dilated_1)
    cv2.imshow('opening_out',opening_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('res_noise1.jpg',opening_out)
    return opening_out

def closing_opening_img():
    dilated = dilation(img)
    eroded = erosion(dilated)
    eroded_1 = erosion(eroded)
    closing_out = dilation(eroded_1)
    cv2.imshow('closing_out',closing_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('res_noise2.jpg',closing_out)
    return closing_out
    
def bound_extraction(a,b):
    k = erosion(a)
    ext_bound1 = np.subtract(a,k)
    cv2.imshow('ext_bound1',ext_bound1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('res_bound1.jpg',ext_bound1)
    l = erosion(b)
    ext_bound2 = np.subtract(b,l)
    cv2.imshow('res_bound2',ext_bound2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('res_bound2.jpg',ext_bound2)
    
def main_func():
    print(' Performing Opening and Closing ')	
    a=opening_closing_img()
    print(' Performing Closing and Opening ')
    b=closing_opening_img()
    print(' Extracting Boundaries ')
    bound_extraction(a,b)
    
main_func()
    
