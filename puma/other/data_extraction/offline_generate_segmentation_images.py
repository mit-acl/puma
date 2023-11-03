#!/usr/bin/env python
# subscribe to T265's raw images and distort it and run it on FastSAM, and get the mean of the blobs
# Author: Kota Kondo
# Atribution: Jouko Kinnari (https://gitlab.com/mit-acl/clipper/uncertain-localization/depth-estimation-experiments/)

import os
import argparse
import cv2
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from snapstack_msgs.msg import Goal
from cv_bridge import CvBridge
import rospy
import rospkg
import numpy as np
from utils import compute_blob_mean_and_covariance, compute_3d_position_of_each_centroid
import motlee_msgs.msg as motlee_msgs
from termcolor import colored
import message_filters
import skimage
from fastsam import FastSAM, FastSAMPrompt
from utils import get_quaternion_from_euler, plotErrorEllipse
import matplotlib.pyplot as plt

# define FastSAM params
conf = 0.5
iou = 0.9
DEVICE = 'cuda'
MAX_BLOB_SIZE = 4000.0
MIN_BLOB_SIZE = 0.0

def main():    

    # convert img to cv2_img
    bridge = CvBridge()
    # load undistorted image
    undistorted_img = cv2.imread("/media/kota/T7/frame/pngs-csvs/ones_used_in_pipeline_diagram/test3-partial/pngs/undistorted_images/frame001500_undistorted.png")
    # undistorted_img = bridge.imgmsg_to_cv2(img, desired_encoding="rgb8")

    # get grayscale image for visualization (segmented images)
    # Let's also make a 1-channel grayscale image
    image_gray = cv2.cvtColor(undistorted_img, cv2.COLOR_RGB2GRAY)
    image_gray_rgb = np.stack((image_gray,)*3, axis=-1)

    # run FastSAM on cv2_img
    rospack = rospkg.RosPack()
    path_to_fastsam = rospack.get_path('puma') + '/scripts/models/FastSAM-x.pt'
    print("path_to_fastsam: ", path_to_fastsam)
    fastSamModel = FastSAM(path_to_fastsam)
    everything_results = fastSamModel(undistorted_img, device=DEVICE, retina_masks=True, imgsz=256, conf=conf, iou=iou,)
    prompt_process = FastSAMPrompt(undistorted_img, everything_results, device=DEVICE)
    segmask = prompt_process.everything_prompt()

    # get centroid of segmask (filtered)
    get_blob_means(segmask, image_gray_rgb, filtered=True)

    # get centroid of segmask (unfiltered)
    get_blob_means(segmask, image_gray_rgb, filtered=False)

# get the blob means and covs
def get_blob_means(segmask, undistorted_img, filtered=True):
    
    # initialize blob means
    blob_means = []
    blob_covs = []

    # If there were segmentations detected by FastSAM, transfer them from GPU to CPU and convert to Numpy arrays
    if (len(segmask) > 0):
        segmask = segmask.cpu().numpy()
    else:
        segmask = None

    if (segmask is not None):
        # FastSAM provides a numMask-channel image in shape C, H, W where each channel in the image is a binary mask
        # of the detected segment
        [numMasks, h, w] = segmask.shape

        # Prepare a mask of IDs where each pixel value corresponds to the mask ID
        segmasks_flat = np.zeros((h,w),dtype=int)

        for maskId in range(numMasks):
            # Extract the single binary mask for this mask id
            mask_this_id = segmask[maskId,:,:]

            # From the pixel coordinates of the mask, compute centroid and a covariance matrix
            blob_mean, blob_cov = compute_blob_mean_and_covariance(mask_this_id)
            
            if filtered:
                # if blob is too small or too big, skip
                if abs(blob_cov[0,0]) > MAX_BLOB_SIZE or abs(blob_cov[1,1]) > MAX_BLOB_SIZE or abs(blob_cov[0,1]) > MAX_BLOB_SIZE \
                    or abs(blob_cov[0,0]) < MIN_BLOB_SIZE or abs(blob_cov[1,1]) < MIN_BLOB_SIZE or abs(blob_cov[0,1]) < MIN_BLOB_SIZE:
                    # print("blob too small or too big")
                    continue

            # Store centroids and covariances in lists
            blob_means.append(blob_mean)
            blob_covs.append(blob_cov)

            # Replace the 1s corresponding with masked areas with maskId (i.e. number of mask)
            segmasks_flat = np.where(mask_this_id < 1, segmasks_flat, maskId)

            # Using skimage, overlay masked images with colors
            undistorted_img = skimage.color.label2rgb(segmasks_flat, undistorted_img)

        # Create a matplotlib figure to plot image and ellipsoids on.
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        for m, c in zip(blob_means, blob_covs):
            plotErrorEllipse(ax, m[0], m[1], c, "r",stdMultiplier=2.0)

        # save image
        ax.imshow(undistorted_img)
        ax.set_xlim(0, undistorted_img.shape[1])
        ax.set_ylim(undistorted_img.shape[0], 0)
        fig.savefig(f"/media/kota/T7/frame/pngs-csvs/ones_used_in_pipeline_diagram/test3-partial/pngs/segmented_images/filtered_segmented_frame1500.png")
        plt.close(fig)

if __name__ == '__main__':
    main()