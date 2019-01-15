import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# Reading the Images
org = cv2.imread('./original_imgs/hough.jpg')
org1 = cv2.imread('./original_imgs/hough.jpg')

org_grayscale = cv2.cvtColor(org, cv2.COLOR_RGB2GRAY)

org_blurred = cv2.GaussianBlur(org_grayscale, (5, 5), 1.5)
img = org_grayscale  

# Defining Sobel Kernels
Gx= np.array([[-1,0,1],
              [-2,0,2],
              [-1,0,1]])
Gy = np.array([[-1,-2,-1],
               [0,0,0],
               [1,2,1]])
k_h = Gx.shape[0]
k_w = Gx.shape[1]
h= Gx.shape[0]//2
w= Gx.shape[1]//2
conv_img= np.zeros((img.shape))

# Performing Convolution for Sobel Operation
def conv(image,b):
    for i in range(h,image.shape[0]-h):
        for j in range(w,image.shape[1]-w):
            s = 0
            for x in range(k_w):
                for y in range(k_h):
                    s = s + (b[x][y]*img[i-h+x][j-w+y])
                    conv_img[i][j]=s                          
    return conv_img
convx = conv(org_blurred,Gx)
convy = conv(org_blurred,Gy)

# magnitude of edges (conbining horizontal and vertical edges)
magnitude = np.sqrt(convx ** 2 + convy ** 2)
magnitude /= np.max(magnitude)

#cv2.namedWindow('edge_magnitude', cv2.WINDOW_NORMAL)
#cv2.imshow('edge_magnitude', magnitude)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
# Combining Gx and Gy
magnitude = np.sqrt(convx ** 2 + convy ** 2)
sobel_img = magnitude
edges = cv2.Canny(org_blurred, 100, 200)
edges1 = cv2.Canny(org,100,200)
for i in range(magnitude.shape[0]):
    for j in range(magnitude.shape[1]):
        if(magnitude[i][j]<100 or magnitude[i][j]>200):
            sobel_img[i][j] = 0
edge = sobel_img 


x1_list = []
x2_list = []
x1_final = []
x2_final = []
x1_list_diag = []
x2_list_diag = []
x1_final_diag = []
x2_final_diag = []
peak_index = []


# Preparing a voting/accumulator for rhos and the angles,a Hough matrix is created
# by running over different angle ranges and obatining rho values
def accumulator(img,deg1,deg2):

    height, width = img.shape
    img_diagonal = np.ceil(np.sqrt(height**2 + width**2)) 
    rhos = np.arange(-img_diagonal, img_diagonal + 1,1)
    angs = np.deg2rad(np.arange(deg1, deg2,1))
    H = np.zeros((len(rhos), len(angs)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img) 

    for i in range(len(x_idxs)): 
        x = x_idxs[i]
        y = y_idxs[i]
        for j in range(len(angs)):
            rho = int((x * np.cos(angs[j]) +
                       y * np.sin(angs[j])) + img_diagonal)
            H[rho,j] = H[rho,j] + 1
    return H, rhos, angs


# Picking the brightest points using line_peak function
def line_peaks(H,peaks,n=7):
    
    indicies = []
    H1 = np.copy(H)
    for i in range(peaks):
        
        # For a particular bright point we bound that point with high pixel values
        # off 255, the bounding ocverage is specified using the variable n
        index = np.argmax(H1) 
        H1_index = np.unravel_index(index, H1.shape) 
        indicies.append(H1_index)
        index_y, index_x = H1_index
        if (index_x - (n/2)) < 0: minimum_x = 0
        else: minimum_x = index_x - (n/2)
        if ((index_x + (n/2) + 1) > H.shape[1]): maximum_x = H.shape[1]
        else: maximum_x = index_x + (n/2) + 1
        if (index_y - (n/2)) < 0:minimum_y = 0
        else:minimum_y = index_y - (n/2)
        if ((index_y + (n/2) + 1) > H.shape[0]): maximum_y = H.shape[0]
        else: maximum_y = index_y + (n/2) + 1
        minimum_x = int(minimum_x)
        minimum_y = int(minimum_y)
        maximum_x = int(maximum_x)
        maximum_y = int(maximum_y)
        for x in range(minimum_x, maximum_x):
            for y in range(minimum_y, maximum_y):
                H1[y, x] = 0
                if (x == minimum_x or x == (maximum_x - 1)):
                       H[y, x] = 255
                if (y == minimum_y or y == (maximum_y - 1)):
                    H[y,x] = 255    
    return indicies, H


# Drawing the vertical lines detected for the image
def lines_vertical(img, indices, rhos, angs):
    
    for i in range(len(indices)):
        rho = rhos[indices[i][0]]
        theta = angs[indices[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))        
        x1_list.append(x1)     
        x2_list.append(x2)
    ind2remove_x1 = [2,6,7,8,10]
    x1_final = [x for i,x in enumerate(x1_list) if i not in ind2remove_x1]
    ind2remove_x2 = [2,6,7,8,10]
    x2_final = [x for i,x in enumerate(x2_list) if i not in ind2remove_x2]
    for i in range(len(x1_final)):
        xk=x1_final[i]
        xj=x2_final[i]
        cv2.line(img, (xk, y1), (xj, y2), (0, 255, 0), 3)

# Drawing the horizontal lines detected for the image
def lines_diagonal(img, indices, rhos, angs2):
    
    for i in range(len(indices)):
        rho = rhos[indices[i][0]]
        theta = angs2[indices[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        x1_list_diag.append(x1)
        x2_list_diag.append(x2)
        cv2.line(img, (x1, y1),(x2, y2), (0, 255, 0), 1)

# Creating Accumulator for Circle Hough Transform
def circle_acc(img,deg1,deg2):
    deg1 = 0
    deg2 = 360
    r = 22
    height, width = img.shape
    thetas = np.deg2rad(np.arange(deg1, deg2, 1))
    H = np.zeros(np.shape(img))
    for i in range(height):
        for j in range(width):
            if(img[i][j]>120):
                for k in range(len(thetas)):
                    a = int(i - r*np.cos(thetas[k]))
                    b = int(j - r*np.sin(thetas[k]))
                    try:
                       H[a,b] += 1
                    except:
                        pass
    return H,thetas

# Picking peaks/bright points from Accumulator
def circle_peaks(H):
    r = 22
    shapes = cv2.imread('./original_imgs/hough.jpg')
    h,w = np.shape(H)
    for i in range(h):
        for j in range(w):
            if(H[i][j] > 150):
                peak_index.append((j,i))
    for k in range(len(peak_index)):
        cv2.circle(shapes,peak_index[k],r,(0,255,0),thickness=3)
    cv2.imwrite('Circle Hough Transform.jpg',shapes)
    cv2.imshow('Final_Image',shapes)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

def main_func():
    
    # FUNCTIONS FOR DETECTING VERTICAL AND DIAGONAL LINES
    print(' **** DETECTING VERTICAL LINES **** ')
    H, rhos, angs = accumulator(edges,deg1=-10,deg2=20)
    print(' **** DETECTING HORIZONTAL LINES **** ')
    H2,rhos2,angs2 = accumulator(edges,deg1 = -40,deg2 = -20)
    peaks_index, H = line_peaks(H, 10, n=7) 
    peaks_index2, H2 = line_peaks(H2,20, n=25)

    lines_diagonal(org,peaks_index2,rhos2,angs2)
    lines_vertical(org1,peaks_index, rhos, angs)
    cv2.namedWindow('Veritcal Hough Transform',cv2.WINDOW_NORMAL)
    cv2.imshow('Veritcal Hough Transform', org1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.namedWindow('Horizontal Hough Transform',cv2.WINDOW_NORMAL)
    cv2.imshow('Horizontal Hough Transform', org)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # FUNCTIONS FOR DETEFCTING CIRCLES
    print(" **** DETECTING CIRCLES **** ")
    Hough_acc,thetas = circle_acc(edges1,0,360)
    circle_peaks(Hough_acc)

main_func()
# Reference AI SHACK Hough Transform and Git hub repositaries
