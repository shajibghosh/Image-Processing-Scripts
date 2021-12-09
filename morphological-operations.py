'''Importing the dependencies'''
import numpy as np
import matplotlib.pyplot as plt

'''Generating Image A and Structuring Element B'''

imageA = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                   [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1],
                   [1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1],
                   [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1],
                   [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1],
                   [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                   [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1],
                   [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
                   [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                   [1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1],
                   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

SE_B = np.array([[1,1,1,1],
                 [1,0,0,1],
                 [1,0,1,1],
                 [1,1,1,1]])

'''Function Definitions'''

def check_index(idx):
    if idx < 0:
        return 0
    else:
        return idx 

def erosion(input_img, SE):
    img_array = np.asarray(input_img)
    SE_array = np.asarray(SE)
    SE_shape = SE.shape
    eroded_img = np.zeros((img_array.shape[0], img_array.shape[1]))
    SE_origin = (int(np.floor((SE_array.shape[0] - 1) / 2.0)), int(np.floor((SE_array.shape[1] - 1) / 2.0)))
    #SE_origin = (int(np.ceil((SE_array.shape[0] - 1) / 2.0)), int(np.ceil((SE_array.shape[1] - 1) / 2.0))) 
    for i in range(len(img_array)):
        for j in range(len(img_array[0])):
            overlap = img_array[check_index(i - SE_origin[0]):i + (SE_shape[0] - SE_origin[0]),
                      check_index(j - SE_origin[1]):j + (SE_shape[1] - SE_origin[1])]
            new_shape = overlap.shape
            SE_first_row_idx = int(np.fabs(i - SE_origin[0])) if i - SE_origin[0] < 0 else 0
            SE_first_col_idx = int(np.fabs(j - SE_origin[1])) if j - SE_origin[1] < 0 else 0

            SE_last_row_idx = SE_shape[0] - 1 - (i + (SE_shape[0] - SE_origin[0]) - img_array.shape[0]) if i + \
            (SE_shape[0] - SE_origin[0]) > img_array.shape[0] else SE_shape[0]-1
            SE_last_col_idx = SE_shape[1] - 1 - (j + (SE_shape[1] - SE_origin[1]) - img_array.shape[1]) if j + \
            (SE_shape[1] - SE_origin[1]) > img_array.shape[1] else SE_shape[1]-1

            if new_shape[0] != 0 and new_shape[1] != 0 and np.array_equal(np.logical_and(overlap, \
                                                                       SE_array[SE_first_row_idx:SE_last_row_idx+1,\
                                                                       SE_first_col_idx:SE_last_col_idx+1]),\
                                                                       SE_array[SE_first_row_idx:SE_last_row_idx+1,
                                                                       SE_first_col_idx:SE_last_col_idx+1]):
                eroded_img[i, j] = 1
    return eroded_img

def dilation(input_img, SE):
    img_array = np.asarray(input_img)
    SE_array = np.asarray(SE)
    SE_shape = SE_array.shape
    dilated_img = np.zeros((img_array.shape[0], img_array.shape[1]))
    SE_origin = ((SE_array.shape[0]-1)//2, (SE_array.shape[1]-1)//2)
    for i in range(len(img_array)):
        for j in range(len(img_array[0])):
            overlap = img_array[check_index(i - SE_origin[0]):i + (SE_shape[0] - SE_origin[0]), \
                                check_index(j - SE_origin[1]):j + (SE_shape[1] - SE_origin[1])]
            new_shape = overlap.shape

            SE_first_row_idx = int(np.fabs(i - SE_origin[0])) if i - SE_origin[0] < 0 else 0
            SE_first_col_idx = int(np.fabs(j - SE_origin[1])) if j - SE_origin[1] < 0 else 0

            SE_last_row_idx = SE_shape[0] - 1 - (i + (SE_shape[0] - SE_origin[0]) - img_array.shape[0]) \
            if i + (SE_shape[0] - SE_origin[0]) > img_array.shape[0] else SE_shape[0]-1
            SE_last_col_idx = SE_shape[1] - 1 - (j + (SE_shape[1] - SE_origin[1]) - img_array.shape[1]) \
            if j + (SE_shape[1] - SE_origin[1]) > img_array.shape[1] else SE_shape[1]-1

            if new_shape[0] != 0 and new_shape[1] != 0 and np.logical_and(SE_array[SE_first_row_idx:SE_last_row_idx+1, \
                                                                        SE_first_col_idx:SE_last_col_idx+1], overlap).any():
                dilated_img[i, j] = 1
    return dilated_img

def opening(input_img, SE):
    im_eroded = erosion(input_img, SE)
    opened_img = dilation(im_eroded, SE)
    return opened_img

def closing(input_img, SE):
    im_dilated = dilation(input_img, SE)
    closed_img = erosion(im_dilated, SE)
    return closed_img

def plot(input_img):
    extent = (0, input_img.shape[0], input_img.shape[1], 0)
    im = plt.imshow(input_img, cmap='gray',extent=extent, interpolation='none', aspect='equal')
    
    ax = plt.gca();
    ax.set_xticks(np.arange(0, input_img.shape[0]+1, 1))
    ax.set_yticks(np.arange(0, input_img.shape[1]+1, 1))

    ax.set_xticklabels(np.arange(0, input_img.shape[0]+1, 1))
    ax.set_yticklabels(np.arange(0, input_img.shape[1]+1, 1))

    ax.set_xticks(np.arange(0, input_img.shape[0]+1, 1), minor=True)
    ax.set_yticks(np.arange(0, input_img.shape[1]+1, 1), minor=True)

    ax.grid(which='major', color='k', linestyle='-', linewidth=1)
    return im 

def main():
    imgEroded = erosion(imageA,SE_B)
    imgDilated = dilation(imageA,SE_B)
    imgOpened = opening(imageA,SE_B)
    imgClosed = closing(imageA,SE_B)
    
    plt.figure(figsize=(16,16))
    plt.subplot(3, 2, 1)
    plot(imageA)
    plt.title('Input Image (A)')
    
    plt.subplot(3, 2, 2)
    plot(SE_B)
    plt.title('Structuring Element (B)')
    
    plt.subplot(3, 2, 3)
    plot(imgEroded)
    plt.title('Eroded Image')
    
    plt.subplot(3, 2, 4)
    plot(imgDilated)
    plt.title('Dilated Image')  
    
    plt.subplot(3, 2, 5)
    plot(imgOpened)
    plt.title('Opening of A by B')
    
    plt.subplot(3, 2, 6)
    plot(imgClosed)
    plt.title('Closing of A by B')
    
    plt.show()

'''Generating Results'''
if __name__ == "__main__":
    main()