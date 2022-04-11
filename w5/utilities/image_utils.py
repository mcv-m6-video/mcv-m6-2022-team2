import os
import cv2
import glob
import random
from tqdm import tqdm
from os.path import dirname, join, exists

def video_to_frames(video_path):
    """
    Read and save a single video from the given path. The frames are saved in the same camera folder in a new folder
    named frames. i.e. is the input is the /S03/c010/vdo.avi this function will save all the in a new folder
    /S03/c010/frames/ which will contain:
    /S03/c010/frames/
            |---0000.jpg
            |---0001.jpg
            |   ...
            |---2174.jpg
            |---2175.jpg
    :param
        video_path: path to the video (../../data/AICity_data/train/S03/c010/vdo.avi)
    """

    # Create the frames' folder inside the camera folder
    camera_path = dirname(video_path)

    # If the folder frames do not exists, it means that the frames have not been extracted
    if not exists(join(camera_path, "frames")):
        os.makedirs(join(camera_path, "frames"), exist_ok=True)

        # Open video capture
        cap = cv2.VideoCapture(video_path)

        # Check if video is opened
        if not cap.isOpened():
            raise IOError("Could not open video")

        # Create the tqdm progress bar object
        frame_num = 1
        pbar = tqdm(desc=f"Reading and saving frames from {camera_path}")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Save frame
            cv2.imwrite(join(camera_path, "frames", f"{frame_num:04d}.jpg"), frame)
            pbar.update(1)
            frame_num += 1

        # Release video
        cap.release()

    else:
        print(f"Frames of {camera_path.split('/')[-2]}/{camera_path.split('/')[-1]} already saved...")


def all_videos_to_frames(data_root="../../data/AICity_data/train"):
    """
    Read and save all the videos from the dataset in individual frames using the video_to_frames function.
    :param
    data_root: path to the dataset (../../data/AICity_data/train)
    """
    # Get a list of all the videos in the dataset
    video_paths = [join(data_root, seq, cam, 'vdo.avi') for seq in os.listdir(data_root) for cam in os.listdir(join(data_root, seq))]
    video_paths = sorted(video_paths)

    # For each video, read and save the frames
    for video in video_paths:
        video_to_frames(video)

def plotBBoxes(img, saveFrames=None, **bboxes):
    """
    Plots a set of bounding boxes on an image.
    parameters:
    ----------------
    img: numpy array of one frame of the sequence
    saveFrames: if not None, saves the frame to the specified path
    bboxes: list of bounding boxes to plot. As the argument is a *args type, several sets of bboxes can be drawn on the same frame.
            i.e., if we wanted to plot the gt and the predicted bboxes, we would call the function as follows:
            plotBBoxes(img, saveFrames, gt_bbox, pred_bbox) where gt_bbox and pred_bbox are a list of the bounding boxes to plot.
    """

    COLORS = [
        (0, 0, 255),
        (0, 255, 0),
        (0, 128, 255),
        (255, 255, 0),
        (255, 0, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 0, 128),
    ]

    for idx, set_bboxes in enumerate(bboxes.values()):
        for bbox in set_bboxes:
            cv2.rectangle(
                img,
                (round(bbox[0]), round(bbox[1])),
                (round(bbox[2]), round(bbox[3])),
                COLORS[idx],
                2,
            )

    if saveFrames is not None:
        cv2.imwrite(saveFrames, img)

    return img

if __name__ == "__main__":
    all_videos_to_frames("../../../data/AICity_data/train")
