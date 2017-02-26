import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import UtilMask

ym_per_pix = 15/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension


class LaneLines:
    """Class to compute lane lines on frames of film."""
    def __init__(self):
        self.fontFace  = cv2.FONT_HERSHEY_SIMPLEX
        self.fontColor = (255, 255, 255)
        self.detected = False
    
    def processFrame(self, topDown, fileName = None):
        """This is the main line finding pipeline function"""
        # mask image
        imgTopMasked = UtilMask.maskPipeline(topDown)
        if fileName:
            mpimg.imsave(os.path.join("test_images/outputs/", fileName+"-3-mask.jpg"), imgTopMasked)
        # compute binary image
        binaryTopDown = UtilMask.binaryImg(imgTopMasked)
        
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binaryTopDown, binaryTopDown, binaryTopDown))*255
        self.imgShape = binaryTopDown.shape
        
        if not self.detected:
            self.blindSearch(binaryTopDown, out_img)
        else:
            found = self.updateLanes(binaryTopDown)
            if not found:
                self.blindSearch(binaryTopDown, out_img)
        
        # if not enough pixels in left or right, re-mask with wider gates
        if len(self.nonzerox[self.lft_lane_inds]) < 300 or  len(self.nonzerox[self.rgt_lane_inds]) < 100:
            imgTopMasked = UtilMask.maskPipeline(topDown, l_thresh=180)
            binaryTopDown = UtilMask.binaryImg(imgTopMasked)
            out_img = np.dstack((binaryTopDown, binaryTopDown, binaryTopDown))*255
            self.blindSearch(binaryTopDown, out_img)
        
        if len(self.nonzerox[self.lft_lane_inds]) > 150 and  len(self.nonzerox[self.rgt_lane_inds]) > 150:
            self.fitLines()
        
        self.highlightLinePoints(out_img)
        return out_img
        
    
    def getLaneFill(self, perspective):
        """Use the given perspective and project a green "safe" lane area onto a blank image"""
        
        # Create an image to draw the lines on
        warp_zero  = np.zeros(self.imgShape).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_lft = np.array([np.transpose(np.vstack([self.lft_fitx, self.ploty]))])
        pts_rgt = np.array([np.flipud(np.transpose(np.vstack([self.rgt_fitx, self.ploty])))])
        pts = np.hstack((pts_lft, pts_rgt))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space
        newwarp = perspective.topDownInv(color_warp)
        return newwarp
    
    def addLaneInfo(self, image):
        """Add annotations for lane curvature and car location"""
        cv2.putText(image, 'Curvature radius  : {:5.2f} km'.format(self.curveRadKm), (20,  60), self.fontFace, 1.5, self.fontColor, 2)
        cv2.putText(image, 'Offset from center: {:5.2f} m '.format(self.laneOffset  ), (20, 110), self.fontFace, 1.5, self.fontColor, 2)
        return

    
    def blindSearch(self, binary_warped, out_img):
        """This function was taken from the lecture notes. Small adaptations to fit my pipeline"""
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[self.imgShape[0]/2:,:], axis=0)
        # Find the peak of the lft and rgt halves of the histogram
        # These will be the starting point for the lft and rgt lines
        midpoint = np.int(histogram.shape[0]/2)
        lftx_base = np.argmax(histogram[:midpoint])
        rgtx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(self.imgShape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        lftx_current = lftx_base
        rgtx_current = rgtx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive lft and rgt lane pixel indices
        self.lft_lane_inds = []
        self.rgt_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and rgt and lft)
            win_y_lo = self.imgShape[0] - (window+1)*window_height
            win_y_hi = self.imgShape[0] - window*window_height
            win_xlft_lo = lftx_current - margin
            win_xlft_hi = lftx_current + margin
            win_xrgt_lo = rgtx_current - margin
            win_xrgt_hi = rgtx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xlft_lo,win_y_lo),(win_xlft_hi,win_y_hi),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xrgt_lo,win_y_lo),(win_xrgt_hi,win_y_hi),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_lft_inds = ((self.nonzeroy >= win_y_lo) & (self.nonzeroy < win_y_hi) & (self.nonzerox >= win_xlft_lo) & (self.nonzerox < win_xlft_hi)).nonzero()[0]
            good_rgt_inds = ((self.nonzeroy >= win_y_lo) & (self.nonzeroy < win_y_hi) & (self.nonzerox >= win_xrgt_lo) & (self.nonzerox < win_xrgt_hi)).nonzero()[0]
            # Append these indices to the lists
            self.lft_lane_inds.append(good_lft_inds)
            self.rgt_lane_inds.append(good_rgt_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_lft_inds) > minpix:
                lftx_current = np.int(np.mean(self.nonzerox[good_lft_inds]))
            if len(good_rgt_inds) > minpix:        
                rgtx_current = np.int(np.mean(self.nonzerox[good_rgt_inds]))

        # Concatenate the arrays of indices
        self.lft_lane_inds = np.concatenate(self.lft_lane_inds)
        self.rgt_lane_inds = np.concatenate(self.rgt_lane_inds)
        
        return out_img
    
    def updateLanes(self, binary_warped):
        """This function was taken from the lecture notes.  Small adaptations to fit my pipeline"""
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero  = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        LFit = self.lft_fit
        RFit = self.rgt_fit
        lft_lane_inds = ((nonzerox > (LFit[0]*(nonzeroy**2) + LFit[1]*nonzeroy + LFit[2] - margin)) & (nonzerox < (LFit[0]*(nonzeroy**2) + LFit[1]*nonzeroy + LFit[2] + margin)))
        rgt_lane_inds = ((nonzerox > (RFit[0]*(nonzeroy**2) + RFit[1]*nonzeroy + RFit[2] - margin)) & (nonzerox < (RFit[0]*(nonzeroy**2) + RFit[1]*nonzeroy + RFit[2] + margin)))
        
        if lft_lane_inds.size == 0 or rgt_lane_inds.size == 0:
            # update lost the lines, lets do blind search
            self.detected = False
            return False
        else:
            self.nonzeroy = nonzeroy
            self.nonzerox = nonzerox
            self.lft_lane_inds = lft_lane_inds
            self.rgt_lane_inds = rgt_lane_inds
            #print("Update", len(self.nonzerox[self.lft_lane_inds]), len(self.nonzerox[self.rgt_lane_inds]))
        return True
    
    def fitLines(self):
        """Take raw left and right pixels and fit lines to them.  Also compute car location
        and lane curvature.  Most of this taken from class notes."""
        
        if self.lft_lane_inds.size == 0 or self.rgt_lane_inds.size == 0:
            self.detected = False
            return
        
        # Extract left and right line pixel positions
        lftx = self.nonzerox[self.lft_lane_inds]
        lfty = self.nonzeroy[self.lft_lane_inds] 
        rgtx = self.nonzerox[self.rgt_lane_inds]
        rgty = self.nonzeroy[self.rgt_lane_inds]
        
        if lftx.size == 0 or lfty.size == 0 or rgtx.size == 0 or rgty.size == 0:
            self.detected = False
            return
        
        # Fit a second order polynomial to each
        self.lft_fit = np.polyfit(lfty, lftx, 2)
        self.rgt_fit = np.polyfit(rgty, rgtx, 2)
    
        # Generate x and y values for plotting
        self.ploty = np.linspace(0, self.imgShape[0]-1, self.imgShape[0] )
        self.lft_fitx = self.lft_fit[0]*self.ploty**2 + self.lft_fit[1]*self.ploty + self.lft_fit[2]
        self.rgt_fitx = self.rgt_fit[0]*self.ploty**2 + self.rgt_fit[1]*self.ploty + self.rgt_fit[2]
        
        # generate lines
        self.lftLine = np.int32(np.stack([self.lft_fitx, self.ploty], axis=1))
        self.rgtLine = np.int32(np.stack([self.rgt_fitx, self.ploty], axis=1))
        
        midPointPx   = 1280.0/2.0
        laneWidthPx  = self.rgt_fitx[-1] - self.lft_fitx[-1]
        laneCenterPx = self.lft_fitx[-1] + laneWidthPx/2.0
        laneOffsetPx = laneCenterPx - midPointPx
        
        self.laneWidth  = laneWidthPx  *xm_per_pix
        self.laneOffset = laneOffsetPx *xm_per_pix
        
        # Fit new polynomials to x,y in world space
        lft_fit_cr = np.polyfit(self.ploty*ym_per_pix, self.lft_fitx*xm_per_pix, 2)
        rgt_fit_cr = np.polyfit(self.ploty*ym_per_pix, self.rgt_fitx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        y_eval = np.max(self.ploty)
        lft_curverad = ((1 + (2*lft_fit_cr[0]*y_eval*ym_per_pix + lft_fit_cr[1])**2)**1.5) / np.absolute(2*lft_fit_cr[0])
        rgt_curverad = ((1 + (2*rgt_fit_cr[0]*y_eval*ym_per_pix + rgt_fit_cr[1])**2)**1.5) / np.absolute(2*rgt_fit_cr[0])
        #print("-- Curvatures: ", lft_curverad, rgt_curverad)
        
        # calculate weighted average
        curveTot = lft_curverad*len(lftx) + rgt_curverad*len(rgtx)
        self.curveRadKm = curveTot/(len(lftx) + len(rgtx))/1000
        #print("  --  Avg Radius = {0:6.2f} km,  Lane Width = {1:.1f} m,   Lane Offset = {2:.1f} m".format(self.curveRadKm, self.laneWidth, self.laneOffset))
        
        self.detected = True
        return
    
    def highlightLinePoints(self, out_img):
        """For debugging, we color the left and right lane raw pixels, and draw the curve fit."""
        out_img[self.nonzeroy[self.lft_lane_inds], self.nonzerox[self.lft_lane_inds]] = [255, 0, 0]
        out_img[self.nonzeroy[self.rgt_lane_inds], self.nonzerox[self.rgt_lane_inds]] = [0, 0, 255]
        
        cv2.polylines(out_img, [self.lftLine, self.rgtLine], False, (255,255,255), thickness=1)
        return out_img
    
    
