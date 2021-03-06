import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Perspective:
    """Class to memorize a top-down perspective transform and apply it when needed."""
    def __init__(self):
        self.isSetup = False
    
    def calcTransform(self, image):
        """On the very first use of this class, this function will generate the 
            perspective transform and save it for future use.
        """
        width  = image.shape[1] # x
        height = image.shape[0] # y
        self.shape = (width, height)
        
        # staring points
        tops1 = [[476, 480], [ 796, 480] ] # top    left,right
        bots1 = [[  0, 680], [1280, 680] ] # bottom left,right
        
        # new point location in top-down image
        tops2 = [[238, 0],      [1038, 0]      ] # top    left,right
        bots2 = [[238, height], [1038, height] ] # bottom left,right
        
        # define transform matrix
        self.source = np.float32([tops1[0], tops1[1], bots1[1], bots1[0]]) # circular order
        self.dest   = np.float32([tops2[0], tops2[1], bots2[1], bots2[0]]) # circular order
    
        self.warpMat    = cv2.getPerspectiveTransform(self.source, self.dest)
        self.warpMatInv = cv2.getPerspectiveTransform(self.dest,   self.source)
        self.isSetup = True
    
    def topDown(self, image):
        if not self.isSetup:
            self.calcTransform(image)
        
        topDownImg = cv2.warpPerspective(image, self.warpMat, self.shape, flags=cv2.INTER_LINEAR)
        return topDownImg
        
    def topDownInv(self, image):
        if not self.isSetup:
            self.calcTransform(image)
        
        topDownImgInv = cv2.warpPerspective(image, self.warpMatInv, self.shape, flags=cv2.INTER_LINEAR)
        return topDownImgInv
    
    def testTransform(self, image):
        if not self.isSetup:
            self.calcTransform(image)
        
        withLines = image.copy()
        cv2.polylines(withLines, [np.int32(self.source)], True,(255,55,255), thickness=2)
        
        topDownWithLines = cv2.warpPerspective(withLines, self.warpMat, self.shape, flags=cv2.INTER_LINEAR)
        return withLines, topDownWithLines
        

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

def binaryImg(image):
    """Extract valid pixels.  Final image will be 1-color"""
    # For now the "blue" channel has the best information from the maskPipeline
    outImg = image[:,:,2]
    return outImg

def maskPipeline(image, ksize=5, \
                    abs_thresh=(50, 200), \
                    mag_thresh=(80, 200), \
                    dir_thresh=(0.7, 1.3), \
                    h_thresh  =(15, 30), \
                    l_thresh  = 210):
    
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
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    #print(hls[56,876,:])
    mask_color = np.zeros_like(s_channel, dtype=np.uint8)
    mask_color[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1]) & (l_channel > 130) & (l_channel < 180) & (s_channel > 100)] = 255
    mask_color[(l_channel > l_thresh)] = 255
    
    combined = np.zeros_like(mask_dir, dtype=np.uint8)
    combined[((abs_mask_x == 1) & (abs_mask_y == 1)) | ((mag_mask == 1) & (mask_dir == 1))] = 255
    
    # mash everything together
    color_binary = np.dstack(( np.zeros_like(combined), combined, mask_color))
        
    return color_binary
    

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

