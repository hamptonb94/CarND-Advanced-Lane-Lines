import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import UtilCamera
import UtilMask
import UtilLines

camera      = UtilCamera.Camera()
perspective = UtilMask.Perspective()
laneLines   = UtilLines.LaneLines()

def imagePipeline(image, fileName=None):
    """Complete process for each frame image.  If 'fileName' is given then each stage
        of the pipeline will write out an image for debugging.
    """
    if fileName:
        mpimg.imsave(os.path.join("test_images/outputs/", fileName), image)
    
    # use camera to un-distort raw frames
    imgUD = camera.undistort(image)
    if fileName:
        mpimg.imsave(os.path.join("test_images/outputs/", fileName+"-0-udist.jpg"), imgUD)
    
    # generate mask from gradients/colors
    imgMasked = UtilMask.maskPipeline(image)
    if fileName:
        mpimg.imsave(os.path.join("test_images/outputs/", fileName+"-1-mask.jpg"), imgMasked)
    
    # change to birds-eye view
    topDown = perspective.topDown(imgUD)
    if fileName:
        mpimg.imsave(os.path.join("test_images/outputs/", fileName+"-2-topdwn.jpg"), topDown)
    
    # output test of perspective to make sure lines are parallel
    if fileName:
        withLines, topDownWithLines = perspective.testTransform(imgUD)
        mpimg.imsave(os.path.join("test_images/outputs/", fileName+"-2a-topdwn2.jpg"), withLines)
        mpimg.imsave(os.path.join("test_images/outputs/", fileName+"-2b-topdwn3.jpg"), topDownWithLines)
    
    # lane pipeline
    searched  = laneLines.processFrame(topDown, fileName)
    if fileName:
        mpimg.imsave(os.path.join("test_images/outputs/", fileName+"-4-search.jpg"), searched)
    
    # get the green safe region
    laneFill = laneLines.getLaneFill(perspective)    
    
    # combine for final result
    imgFinal = UtilMask.weighted_img(laneFill, imgUD, α=0.8, β=0.3)
    if fileName:
        mpimg.imsave(os.path.join("test_images/outputs/", fileName+"-5-final.jpg"), imgFinal)
    
    # also add a test overlay for curvature and car location
    laneLines.addLaneInfo(imgFinal)
    if fileName:
        mpimg.imsave(os.path.join("test_images/outputs/", fileName+"-6-annot.jpg"), imgFinal)
        
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
    global laneLines
    makeNewDir("test_images/outputs/")
    fileNames = os.listdir("test_images/")
    for fileName in fileNames:
        if 'jpg' not in fileName:
            continue
        print("Processing: ", fileName)
        fullName = os.path.join("test_images",fileName)
        image = mpimg.imread(fullName)
        laneLines = UtilLines.LaneLines() # reset lane lines for test images
        imagePipeline(image, fileName)
        
from moviepy.editor import VideoFileClip

def processMovie(movieName):
    outputName = 'out-'+movieName
    clip1 = VideoFileClip(movieName)
    out_clip = clip1.fl_image(imagePipeline) #NOTE: this function expects color images!!
    out_clip.write_videofile(outputName, audio=False)

import sys
if __name__ == '__main__':
    """Main processing script.
        If no arguments given, it will process all images in the test_images folder.  
        A single argument is the name of a movie to process.
    """
    if len(sys.argv) > 1:
        processMovie(sys.argv[1])
    else:
        processImages()
