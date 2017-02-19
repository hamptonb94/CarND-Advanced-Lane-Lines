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
    imgUD = cam.undistort(image)
    if fileName:
        mpimg.imsave(os.path.join("test_images/outputs/", fileName+"-0-udist.jpg"), imgUD)
    
    imgMasked = UtilMask.maskPipeline(image)
    if fileName:
        mpimg.imsave(os.path.join("test_images/outputs/", fileName+"-1-mask.jpg"), imgMasked)
    
    # birds-eye
    

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
        mpimg.imsave(os.path.join("test_images/outputs/", fileName), image)
        
        imagePipeline(image, fileName)
        

def processMovie1():
    print("Movie 1")


processImages()
processMovie1()
