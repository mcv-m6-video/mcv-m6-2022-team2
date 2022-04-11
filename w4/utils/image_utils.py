import cv2
import os
from tqdm import tqdm

DDBB_ROOT = "../../data/AICity_data/train/"


def read_and_save_video(video_path):
    """
    Read and save video from given path.
    The frames are saved in the w4/frames/ folder with the same structure as the AICity dataset
    i.e.
    w4/frames/
            |---S01/
            |       |---c001/
            |          ...
            |       |---c005/
            |---S03/
            |       |---c010/
            |          ...
            |       |---c015/
            |---S04/
                    |---c016/
                      ...
                    |---c040/
    parameters:
    ----------------
    video_path: path to the video (../../data/AICity_data/train/S03/c010/vdo.avi)
    """

    # If the folder frames does not exist, create it
    os.makedirs(os.path.join(os.getcwd(), "frames"), exist_ok=True)

    # If the folder of the corresponding sequence does not exist, create it
    sequence_name = video_path.split("/")[-3]
    os.makedirs(os.path.join(os.getcwd(), "frames", sequence_name), exist_ok=True)

    # If the folder of the corresponding video does not exist, create it as well as write the frames
    cam_name = video_path.split("/")[-2]

    if not os.path.exists(os.path.join(os.getcwd(), "frames", sequence_name, cam_name)):
        os.makedirs(
            os.path.join(os.getcwd(), "frames", sequence_name, cam_name), exist_ok=True
        )

        # Copy the ground truth txt to the corresponding frame folder
        gt_path = os.path.join(video_path[:-8], "gt", "gt.txt")
        os.system(
            f"cp {gt_path} {os.path.join(os.getcwd(), 'frames', sequence_name, cam_name, 'gt.txt')}"
        )

        cap = cv2.VideoCapture(video_path)

        # Check if video is opened
        if not cap.isOpened():
            raise IOError("Could not open video")

        frame_num = 1
        pbar = tqdm(desc=f"Reading and saving frames from {sequence_name}/{cam_name}")

        # Read video
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Save frame
            cv2.imwrite(
                os.path.join(
                    os.getcwd(),
                    "frames",
                    sequence_name,
                    cam_name,
                    f"{frame_num:04}.jpg",
                ),
                frame,
            )
            pbar.update(1)
            frame_num += 1

        # Release video
        cap.release()

    else:
        print(f"Frames already saved for {sequence_name}/{cam_name}")


def read_txt_save_videos(txt_path):
    """
    Reads a txt file with the path to the videos and saves the corresponding frames.
    The txt file contains the name of the sequences which correspond to the train/test set. It must have the following format:
    ----------------
    S03/c010/
    S03/c011/
    S03/c012/
    S03/c013/
    S03/c014/
    S03/c015/
    ----------------
    parameters:
    ----------------
    txt_path: path to the txt file with the paths to the videos
    """
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            video_path = os.path.join(DDBB_ROOT, line, "vdo.avi")
            read_and_save_video(video_path)


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
