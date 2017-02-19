import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define constants for region masking images
LANE_TOP = 330.
LANE_BOT =  56.
INSET_LT = 150.
INSET_RT =  50.
INSET_MI = 420.

LANE_TOP_R = LANE_TOP/540
LANE_BOT_R = LANE_BOT/720
INSET_LT_R = INSET_LT/960
INSET_RT_R = INSET_RT/960
INSET_MI_R = INSET_MI/960

def region_of_interest(img):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    width  = img.shape[1] # x
    height = img.shape[0] # y
    
    insetTOP= int(LANE_TOP_R * height)
    insetBOT= int(LANE_BOT_R * height)
    insetLT = int(INSET_LT_R * width)
    insetRT = int(INSET_RT_R * width)
    insetMI = int(INSET_MI_R * width)
    
    botLeft = (insetLT,height-insetBOT)
    topLeft = (insetMI,insetTOP)
    topRight= (width-insetMI,insetTOP)
    botRight= (width-insetRT,height-insetBOT)
    vertices = np.array([[botLeft, topLeft, topRight, botRight]], dtype=np.int32)
    
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def maskPipeline(image, ksize=5, \
                    abs_thresh=(50, 200), \
                    mag_thresh=(80, 200), \
                    dir_thresh=(0.7, 1.3), \
                    s_thresh  =(170, 255)):
    
    """Compute total image mask based on Sobel gradients and color values"""
    
    # compute Sobels
    imgGray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(imgGray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(imgGray, cv2.CV_64F, 0, 1, ksize=ksize)
    
    # 3) Take the absolute value of the derivative or gradient
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))
    
    # abs-threshold mask
    abs_mask_x = np.zeros_like(scaled_sobelx)
    abs_mask_y = np.zeros_like(scaled_sobely)
    abs_mask_x[(scaled_sobelx >= abs_thresh[0]) & (scaled_sobelx <= abs_thresh[1])] = 1
    abs_mask_y[(scaled_sobely >= abs_thresh[0]) & (scaled_sobely <= abs_thresh[1])] = 1
    
    # magnitude mask
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*sobel_mag/np.max(sobel_mag))
    mag_mask = np.zeros_like(scaled_sobel)
    mag_mask[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    
    # direction mask
    dir_sobel = np.arctan2(abs_sobely, abs_sobelx)
    mask_dir = np.zeros_like(dir_sobel)
    mask_dir[(dir_sobel >= dir_thresh[0]) & (dir_sobel <= dir_thresh[1])] = 1
    
    # color mask
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    mask_col = np.zeros_like(s_channel, dtype=np.uint8)
    mask_col[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 255
    
    combined = np.zeros_like(mask_dir, dtype=np.uint8)
    combined[((abs_mask_x == 1) & (abs_mask_y == 1)) | ((mag_mask == 1) & (mask_dir == 1))] = 255
    
    # mash everything together
    color_binary = np.dstack(( np.zeros_like(combined), combined, mask_col))
    
    masked = region_of_interest(color_binary)
    
    return masked
    

if __name__ == '__main__':
    fileNames = os.listdir("test_images/")
    for fileName in fileNames:
        if 'jpg' not in fileName:
            continue
        print("Processing: ", fileName)
        fullName = os.path.join("test_images",fileName)
        image = mpimg.imread(fullName)
        result = maskPipeline(image)
        mpimg.imsave(os.path.join("test_images/outputs/", fileName+",mask.jpg"), result)
        final = weighted_img(result, image)
        mpimg.imsave(os.path.join("test_images/outputs/", fileName+",final.jpg"), final)
