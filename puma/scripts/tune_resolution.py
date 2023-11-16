#!/usr/bin/env python
# undistort images

import os
import argparse
import cv2
import numpy as np
import rosbag
from PIL import Image
from FastSAM.fastsam import *
import skimage
from utils import compute_blob_mean_and_covariance, compute_3d_position_of_each_centroid, plotErrorEllipse
import matplotlib.pyplot as plt
import time

### params
conf = 0.5
iou = 0.9
# DEVICE = 'cuda'
DEVICE = 'cpu'
MAX_BLOB_SIZE = 500.0
MIN_BLOB_SIZE = 0.0

fastSamModel = FastSAM('./FastSAM/Models/FastSAM-x.pt')

def save_segmented_image(segmask, undistorted_img, output_dir, filename, factor):

    # initialize blob means and covariances
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
            
            # if blob is too small or too big, skip
            if abs(blob_cov[0,0]) > MAX_BLOB_SIZE or abs(blob_cov[1,1]) > MAX_BLOB_SIZE or abs(blob_cov[0,1]) > MAX_BLOB_SIZE \
                or abs(blob_cov[0,0]) < MIN_BLOB_SIZE or abs(blob_cov[1,1]) < MIN_BLOB_SIZE or abs(blob_cov[0,1]) < MIN_BLOB_SIZE:
                # print("blob too small or too big")
                continue

            # Store centroids and covariances in lists
            # blob_means.append(blob_mean)
            # blob_covs.append(blob_cov)

            # Replace the 1s corresponding with masked areas with maskId (i.e. number of mask)
            segmasks_flat = np.where(mask_this_id < 1, segmasks_flat, maskId)

        # Using skimage, overlay masked images with colors
        segmented_img = skimage.color.label2rgb(segmasks_flat, undistorted_img)

        # Create a matplotlib figure to plot image and ellipsoids on.
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        # plot an ellipse
        for m, c in zip(blob_means, blob_covs):
            plotErrorEllipse(ax, m[0], m[1], c, "r",stdMultiplier=2.0)

        # show the image
        ax.imshow(segmented_img)
        ax.set_xlim([0,w])
        ax.set_ylim([h,0])        

        # Save segmented image
        output_filename = filename[:-4] + f'_segmented_factor_{round(factor,2)}.png'
        fig.savefig(os.path.join(output_dir, 'segmented', output_filename))
        # cv2.imwrite(os.path.join(output_dir, 'segmented', output_filename), segmented_img)
        plt.close(fig)
        

def undistort_images_rs_resolution_test(input_dir, output_dir, K, D, R, P):

    # list images
    img_list = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            img_list.append(filename)
    img_list.sort()

    # undistort images
    for filename in img_list:
        img = cv2.imread(os.path.join(input_dir, filename))
        h,  w = img.shape[:2]
        size_scale = 1.0
        balance = 1.0
        img_dim_out =(int(w*size_scale), int(h*size_scale))

        # OpenCV fishey calibration cuts too much of the resulting image - https://stackoverflow.com/questions/34316306/opencv-fisheye-calibration-cuts-too-much-of-the-resulting-image
        # so instead of P, we use scaled K (nK)
        scaling_factor = 0.3 # 0.3 is most resonable based on tune_scaling_factor.py
        nK = K.copy()
        nK[0,0] = K[0,0] * scaling_factor
        nK[1,1] = K[1,1] * scaling_factor

        for resolution_factor in range(0, 85, 10):

            print()
            print("**************************************")
            print("resolution_factor: ", resolution_factor)
            print("**************************************")
            print()
            
            # map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, R, P, img_dim_out, cv2.CV_32FC1)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, R, nK, img_dim_out, cv2.CV_32FC1)
            undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            # blurred_img = cv2.GaussianBlur(undistorted_img, (resolution_factor, resolution_factor), 0)
            blurred_img = cv2.GaussianBlur(undistorted_img, (resolution_factor, resolution_factor), 0)
            output_filename = filename[:-4] + f'_undistorted_res_{resolution_factor}.png'
            cv2.imwrite(os.path.join(output_dir, 'undistorted', output_filename), blurred_img)

            #run FastSAM on cv2_img
            everything_results = fastSamModel(blurred_img, device=DEVICE, retina_masks=True, imgsz=1024, conf=conf, iou=iou,)
            prompt_process = FastSAMPrompt(blurred_img, everything_results, device=DEVICE)
            segmask = prompt_process.everything_prompt()

            # segment image and save it
            save_segmented_image(segmask, blurred_img, output_dir, filename, resolution_factor)

def undistort_images_rs_resize_test(input_dir, output_dir, K, D, R, P):

    # list images
    img_list = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            img_list.append(filename)
    img_list.sort()

    # undistort images
    for filename in img_list:
        img = cv2.imread(os.path.join(input_dir, filename))
        h,  w = img.shape[:2]
        img_dim_out =(int(w), int(h))

        # OpenCV fishey calibration cuts too much of the resulting image - https://stackoverflow.com/questions/34316306/opencv-fisheye-calibration-cuts-too-much-of-the-resulting-image
        # so instead of P, we use scaled K (nK)
        scaling_factor = 0.3 # 0.3 is most resonable based on tune_scaling_factor.py
        nK = K.copy()
        nK[0,0] = K[0,0] * scaling_factor
        nK[1,1] = K[1,1] * scaling_factor

        for resize_factor in np.arange(0.1, 1.0, 0.1):

            print()
            print("**************************************")
            print("resize_factor: ", resize_factor)
            print("**************************************")
            print()
            
            # map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, R, P, img_dim_out, cv2.CV_32FC1)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, R, nK, img_dim_out, cv2.CV_32FC1)
            undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            resized_img = cv2.resize(undistorted_img, (int(resize_factor*w), int(resize_factor*h)), interpolation = cv2.INTER_AREA)
            output_filename = filename[:-4] + f'_undistorted_resize_{round(resize_factor,2)}.png'
            cv2.imwrite(os.path.join(output_dir, 'undistorted', output_filename), resized_img)

            #run FastSAM on cv2_img
            everything_results = fastSamModel(resized_img, device=DEVICE, retina_masks=True, imgsz=1024, conf=conf, iou=iou,)
            prompt_process = FastSAMPrompt(resized_img, everything_results, device=DEVICE)
            segmask = prompt_process.everything_prompt()

            # segment image and save it
            save_segmented_image(segmask, resized_img, output_dir, filename, resize_factor)

def undistort_images_rs_resize_test2(input_dir, output_dir, K, D, R, P):

    # list images
    img_list = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            img_list.append(filename)
    img_list.sort()

    # undistort images
    for filename in img_list:
        img = cv2.imread(os.path.join(input_dir, filename))
        h,  w = img.shape[:2]
        img_dim_out =(int(w), int(h))

        # OpenCV fishey calibration cuts too much of the resulting image - https://stackoverflow.com/questions/34316306/opencv-fisheye-calibration-cuts-too-much-of-the-resulting-image
        # so instead of P, we use scaled K (nK)
        scaling_factor = 0.3 # 0.3 is most resonable based on tune_scaling_factor.py
        nK = K.copy()
        nK[0,0] = K[0,0] * scaling_factor
        nK[1,1] = K[1,1] * scaling_factor

        # calculate computation time
        # note: the first run is always slower, so we run it once before the loop
        initialized = False
        computation_times = {}

        for imgsz in np.arange(128, 1024+32, 32):
            imgsz = int(imgsz)
            print()
            print("**************************************")
            print("imgsz: ", imgsz)
            print("**************************************")
            print()
            
            # map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, R, P, img_dim_out, cv2.CV_32FC1)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, R, nK, img_dim_out, cv2.CV_32FC1)
            undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            output_filename = filename[:-4] + f'_undistorted_resize_{round(imgsz,2)}.png'
            cv2.imwrite(os.path.join(output_dir, 'undistorted', output_filename), undistorted_img)

            #run FastSAM on cv2_img
            if not initialized:
                everything_results = fastSamModel(undistorted_img, device=DEVICE, retina_masks=True, imgsz=imgsz, conf=conf, iou=iou,)
                initialized = True

            start_time = time.time()
            everything_results = fastSamModel(undistorted_img, device=DEVICE, retina_masks=True, imgsz=imgsz, conf=conf, iou=iou,)
            computation_times[str(imgsz)] = (time.time() - start_time) * 1000
            
            prompt_process = FastSAMPrompt(undistorted_img, everything_results, device=DEVICE)
            segmask = prompt_process.everything_prompt()

            # segment image and save it
            save_segmented_image(segmask, undistorted_img, output_dir, filename, imgsz)

    return computation_times

def main():
    """Undistort images"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Undistort images.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("test_dir", help="Test Directory.")
    args = parser.parse_args()

    # check if output folder exists and create it if not
    input_dir = os.path.join(args.test_dir, "pngs/raw_images")
    output_dir = os.path.join(args.test_dir, "pngs/resolution_test")

    # get camera matrix and distortion coefficients 
    # for realsense
    bag = rosbag.Bag(args.bag_file, "r")
    t265_fisheye1_camera_info_topic = "/t265/fisheye1/camera_info"
    for topic, msg, t in bag.read_messages(topics=[t265_fisheye1_camera_info_topic]):
        # save K_fe1, D_fe1
        if topic == t265_fisheye1_camera_info_topic:
            K_fe1 = np.array(msg.K).reshape(3,3)
            D_fe1 = np.array(msg.D)
            R_fe1 = np.array(msg.R).reshape(3,3)
            P_fe1 = np.array(msg.P).reshape(3,4)
            break
    
    print("K_fe1: ", K_fe1)
    print("D_fe1: ", D_fe1)
    print("R_fe1: ", R_fe1)
    print("P_fe1: ", P_fe1)
    
    # undistort images
    # undistort_images_rs_resolution_test(os.path.join(input_dir, "t265_fisheye1"), os.path.join(output_dir, "t265_fisheye1/blurring"), K_fe1, D_fe1, R_fe1, P_fe1)
    # undistort_images_rs_resize_test(os.path.join(input_dir, "t265_fisheye1"), os.path.join(output_dir, "t265_fisheye1/resizing"), K_fe1, D_fe1, R_fe1, P_fe1)
    
    computation_times = undistort_images_rs_resize_test2(os.path.join(input_dir, "t265_fisheye1"), os.path.join(output_dir, "t265_fisheye1/resizing2"), K_fe1, D_fe1, R_fe1, P_fe1)

    # plot computation times
    plt.figure()
    plt.plot(computation_times.keys(), computation_times.values())
    plt.xlabel("image size")
    plt.ylabel("computation time (ms)")
    plt.title("Computation time vs. image size on CPU")
    plt.grid()
    plt.xticks(rotation='vertical')
    plt.yticks(np.arange(0, 2200, 100))
    plt.savefig(os.path.join(output_dir, "t265_fisheye1/resizing2/computation_time_vs_image_size.png"))


    return

if __name__ == '__main__':
    main()