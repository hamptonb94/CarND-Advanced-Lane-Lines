import numpy as np
import cv2
import glob
import os.path
import pickle
import matplotlib.pyplot as plt

CALIBRATION_FILE = 'camera_calibration.p'

class Camera:
    def __init__(self):
        if os.path.isfile(CALIBRATION_FILE):
            # load calibration
            cam_pickle = pickle.load(open(CALIBRATION_FILE, "rb"))
            self.cameraMatrix = cam_pickle['mtx']
            self.distCoeffs   = cam_pickle['dist']
        else:
            self.calibrateCamera()
    
    def calibrateCamera(self):
        # index array
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
        
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.
        
        # Make a list of calibration images
        images = glob.glob('camera_cal/cal*.jpg')
        
        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (9,6), corners, ret)
                write_name = 'tmp/corners_found'+str(idx)+'.jpg'
                cv2.imwrite(write_name, img)
    
        # Do camera calibration given object points and image points
        img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
        self.cameraMatrix = mtx
        self.distCoeffs   = dist
        
        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        cam_pickle = {'mtx':mtx, 'dist':dist}
        pickle.dump( cam_pickle, open(CALIBRATION_FILE, "wb") )
    
    def undistort(self, img):
        return cv2.undistort(img, self.cameraMatrix, self.distCoeffs, None, self.cameraMatrix)
    
    def writeTest(self):
        images = glob.glob('camera_cal/calibration1.jpg')
        img = cv2.imread(images[0])
        dst = self.undistort(img)
        cv2.imwrite('output_images/calibration1_undist.jpg', dst)

if __name__ == '__main__':
    cam = Camera()
    cam.writeTest()
