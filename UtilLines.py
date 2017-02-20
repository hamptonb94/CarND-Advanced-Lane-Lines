import numpy as np
import cv2
import matplotlib.pyplot as plt

ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension


class LaneLines:
    
    def getLaneFill(self, binary_warped, perspective):
        # Create an image to draw the lines on
        warp_zero  = np.zeros_like(binary_warped).astype(np.uint8)
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
    
    def blindSearch(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the lft and rgt halves of the histogram
        # These will be the starting point for the lft and rgt lines
        midpoint = np.int(histogram.shape[0]/2)
        lftx_base = np.argmax(histogram[:midpoint])
        rgtx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        lftx_current = lftx_base
        rgtx_current = rgtx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive lft and rgt lane pixel indices
        lft_lane_inds = []
        rgt_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and rgt and lft)
            win_y_lo = binary_warped.shape[0] - (window+1)*window_height
            win_y_hi = binary_warped.shape[0] - window*window_height
            win_xlft_lo = lftx_current - margin
            win_xlft_hi = lftx_current + margin
            win_xrgt_lo = rgtx_current - margin
            win_xrgt_hi = rgtx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xlft_lo,win_y_lo),(win_xlft_hi,win_y_hi),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xrgt_lo,win_y_lo),(win_xrgt_hi,win_y_hi),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_lft_inds = ((nonzeroy >= win_y_lo) & (nonzeroy < win_y_hi) & (nonzerox >= win_xlft_lo) & (nonzerox < win_xlft_hi)).nonzero()[0]
            good_rgt_inds = ((nonzeroy >= win_y_lo) & (nonzeroy < win_y_hi) & (nonzerox >= win_xrgt_lo) & (nonzerox < win_xrgt_hi)).nonzero()[0]
            # Append these indices to the lists
            lft_lane_inds.append(good_lft_inds)
            rgt_lane_inds.append(good_rgt_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_lft_inds) > minpix:
                lftx_current = np.int(np.mean(nonzerox[good_lft_inds]))
            if len(good_rgt_inds) > minpix:        
                rgtx_current = np.int(np.mean(nonzerox[good_rgt_inds]))

        # Concatenate the arrays of indices
        lft_lane_inds = np.concatenate(lft_lane_inds)
        rgt_lane_inds = np.concatenate(rgt_lane_inds)

        # Extract lft and rgt line pixel positions
        lftx = nonzerox[lft_lane_inds]
        lfty = nonzeroy[lft_lane_inds] 
        rgtx = nonzerox[rgt_lane_inds]
        rgty = nonzeroy[rgt_lane_inds] 

        # Fit a second order polynomial to each
        lft_fit = np.polyfit(lfty, lftx, 2)
        rgt_fit = np.polyfit(rgty, rgtx, 2)
    
    
        # Generate x and y values for plotting
        self.ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        self.lft_fitx = lft_fit[0]*self.ploty**2 + lft_fit[1]*self.ploty + lft_fit[2]
        self.rgt_fitx = rgt_fit[0]*self.ploty**2 + rgt_fit[1]*self.ploty + rgt_fit[2]
        
        midPointPx   = 1280.0/2.0
        laneWidthPx  = self.rgt_fitx[-1] - self.lft_fitx[-1]
        laneCenterPx = self.lft_fitx[-1] + laneWidthPx/2.0
        laneOffsetPx = laneCenterPx - midPointPx
        
        self.laneWidth  = laneWidthPx  *xm_per_pix
        self.laneOffset = laneOffsetPx *xm_per_pix
        
        out_img[nonzeroy[lft_lane_inds], nonzerox[lft_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[rgt_lane_inds], nonzerox[rgt_lane_inds]] = [0, 0, 255]
    
        # Fit new polynomials to x,y in world space
        lft_fit_cr = np.polyfit(self.ploty*ym_per_pix, self.lft_fitx*xm_per_pix, 2)
        rgt_fit_cr = np.polyfit(self.ploty*ym_per_pix, self.rgt_fitx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        y_eval = np.max(self.ploty)
        lft_curverad = ((1 + (2*lft_fit_cr[0]*y_eval*ym_per_pix + lft_fit_cr[1])**2)**1.5) / np.absolute(2*lft_fit_cr[0])
        rgt_curverad = ((1 + (2*rgt_fit_cr[0]*y_eval*ym_per_pix + rgt_fit_cr[1])**2)**1.5) / np.absolute(2*rgt_fit_cr[0])
        # Now our radius of curvature is in meters
        #print(lft_curverad, 'm', rgt_curverad, 'm')
    
        # calculate weighted average
        curveTot = lft_curverad*len(lftx) + rgt_curverad*len(rgtx)
        curve_radius = curveTot/(len(lftx) + len(rgtx))
        
        print("  --  Avg Radius = {0:8.1f} m,  Lane Width = {1:.1f} m,   Lane Offset = {2:.1f} m".format(curve_radius, self.laneWidth, self.laneOffset))
    
        return out_img
