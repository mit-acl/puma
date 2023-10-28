#!/usr/bin/env python
# Author: Kota Kondo

import cv2
import os
import argparse

def main():

    # parse command line arguments
    parser = argparse.ArgumentParser(description='Make video from pngs')
    parser.add_argument("-d", "--input_dir", help="Input directory.", default="/media/kota/T7/frame/sim/benchmarking/ones_used_in_icra_paper/videos")
    args = parser.parse_args()


    # list all the undistorted images
    test_folders = os.listdir(args.input_dir)
    test_folders.sort()
    
    for test_folder in test_folders:

        print("test_folder: ", test_folder)

        # if it's not directory, skip
        if not os.path.isdir(os.path.join(args.input_dir, test_folder)):
            continue
    
        case_subfolders = os.listdir(os.path.join(args.input_dir, test_folder))
        case_subfolders.sort()

        for case_subfolder in case_subfolders:

            print("case_subfolder: ", case_subfolder)

            # if it's not directory, skip
            if not os.path.isdir(os.path.join(args.input_dir, test_folder, case_subfolder)):
                continue

            veh_subfolders = os.listdir(os.path.join(args.input_dir, test_folder, case_subfolder, "data/pngs/segmented_filtered_images"))
            veh_subfolders.sort()

            for veh_subfolder in veh_subfolders:

                print("veh_subfolder: ", veh_subfolder)

                # get the segmented images
                image_folder = os.path.join(args.input_dir, test_folder, case_subfolder, "data/pngs/segmented_filtered_images", veh_subfolder, "t265_fisheye1")
                segmented_imgs = os.listdir(image_folder)
                segmented_imgs.sort()

                # get the first image to get the size of the video
                frame = cv2.imread(os.path.join(image_folder, segmented_imgs[0]))
                height, width, layers = frame.shape

                # create a video writer (https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html#ad59c61d8881ba2b2da22cff5487465b5)
                print("creating video...")
                total_frames = len(segmented_imgs)

                video_name = os.path.join(args.input_dir, test_folder, case_subfolder, f'downward-camera-{test_folder}-{veh_subfolder}.mp4')
                video = cv2.VideoWriter(filename=video_name, fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=total_frames/100, frameSize=(width,height), isColor=True)

                for image in segmented_imgs:
                    video.write(cv2.imread(os.path.join(image_folder, image)))

                # release the video writer
                video.release()
                cv2.destroyAllWindows()

if __name__ == '__main__':
    main()