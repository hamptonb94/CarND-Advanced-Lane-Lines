import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import UtilCamera
import UtilMask

cam = UtilCamera.Camera()

def imagePipeline(image, fileName=None):
    if fileName:
        mpimg.imsave(os.path.join("test_images/outputs/", fileName), image)
    
    imgUD = cam.undistort(image)
    if fileName:
        mpimg.imsave(os.path.join("test_images/outputs/", fileName+"-0-udist.jpg"), imgUD)
    
    imgMasked = UtilMask.maskPipeline(image)
    if fileName:
        mpimg.imsave(os.path.join("test_images/outputs/", fileName+"-1-mask.jpg"), imgMasked)
    
    # birds-eye
    topDown = UtilMask.topDown(imgUD)
    if fileName:
        mpimg.imsave(os.path.join("test_images/outputs/", fileName+"-2-topdwn.jpg"), topDown)
    
    # masking
    imgTopMasked = UtilMask.maskPipeline(topDown)
    if fileName:
        mpimg.imsave(os.path.join("test_images/outputs/", fileName+"-3-mask.jpg"), imgTopMasked)
    
    # combine for final result
    imgFinal = UtilMask.weighted_img(imgMasked, imgUD)
    if fileName:
        mpimg.imsave(os.path.join("test_images/outputs/", fileName+"-5-final.jpg"), imgFinal)
    
    return imgFinal
    

def makeNewDir(dir):
    if not os.path.isdir(dir):
        try:
            os.mkdir(dir)
        except:
            raise Exception("Error, %s is not a valid output directory\n" %dir)
    if not os.path.isdir(dir):
        raise Exception("Error, could not make run directory %s, exiting" %dir)


def processImages():
    makeNewDir("test_images/outputs/")
    fileNames = os.listdir("test_images/")
    for fileName in fileNames:
        if 'jpg' not in fileName:
            continue
        print("Processing: ", fileName)
        fullName = os.path.join("test_images",fileName)
        image = mpimg.imread(fullName)
        imagePipeline(image, fileName)
        
from moviepy.editor import VideoFileClip

def processMovie1():
    print("Movie 1")
    white_output = 'video_output.mp4'
    clip1 = VideoFileClip("challenge_video.mp4")
    white_clip = clip1.fl_image(imagePipeline) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)


processImages()
#processMovie1()
