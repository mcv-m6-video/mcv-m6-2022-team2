import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from metric_functions import mean_iou
import glob
from PIL import Image

def plot_precision_recall_one_class(prec, recall, ap, info):
    """
    Plot precision and recall
    :param prec
    :param recall
    :param ap
    """
    plt.plot(prec, recall)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision vs recall in model {info} with AP: {ap}')
    plt.show()


def plot_iou_vs_time(det_file_path, frames_paths, ground_truth, detections):
    """
    Plot iou vs time
    (params are explained in other functions)
    :param det_file_path
    :param frames_paths
    :param ground_truth
    :param detections
    """
    miou = np.empty(0, )
    idFrames = np.array(())
    for frame in frames_paths:
        if os.name == 'nt':
            frame = frame.replace(os.sep, '/')
        frame_id = (frame.split('/')[-1]).split('.')[0]

        idFrames = np.append(idFrames, int(frame_id))
        gt_frame = np.array(dict_to_list(ground_truth[frame_id]))
        dets_frame = np.array(dict_to_list(detections[frame_id]))

        mean = mean_iou(gt_frame, dets_frame)
        miou = np.hstack((miou, mean))

    print(f'Mean Iou: {np.round(np.mean(miou),4)}')
    plt.figure()
    for i in range(800,1000):
        plt.plot(idFrames[:i], miou[:i], color="blue")
        plt.ylim([0, 1])
        plt.xlim([800, 1000])
        plt.xlabel('Frame')
        plt.ylabel('mean IoU')
        plt.title(f'Mean IoU in function of the frame in model {det_file_path.split(".txt")[0]}')
        plt.pause(0.05)
        
        """ if not os.path.exists(det_file_path.split(".txt")[0] + '_meanIoU'):
            os.mkdir(det_file_path.split(".txt")[0] + '_meanIoU')
            
        if i < 1000:
            frame = '0' + str(i) 
        else:
            frame = str(i)
            
        plt.savefig(f'{det_file_path.split(".txt")[0]}_meanIoU/meanIou_{frame}.png') """
    
    plt.show()


def dict_to_list(frame_info):
    """
    Transform the bbox that is in the dictionary into a list
    :param frame_info: dictionary with the information needed to create the list
    :return: return the list created
    """
    return [[obj['bbox'][0], obj['bbox'][1], obj['bbox'][2], obj['bbox'][3]] for obj in frame_info]

def plotBBox(img_path, initalFrame, finalFrame, modelName, saveFrames=False, **labels):
    """
    Plots bounding boxes into the selected frames
    :param img_path: path to all the images
    :param initalFrame: frame to start plotting bounding bboxes
    :param finalFrame: frame to end plotting bboxes
    :param saveFrames: if it is set to true, stores frames in dir /frames_stored and generates gifs
    :param labels: the idea is to be a dict with gt and detections or only gt or detections
    :return: frames with bboxes plotted in it
    """

    frames = []
    COLORS=[(0,255,0), (0,0,255)]

    print('plotting bounding boxes in the frames...')
    for frame_num in tqdm(range(initalFrame, finalFrame)):
        im = cv2.imread(img_path[frame_num])
        for idx, (name, labels_total) in enumerate(labels.items()):
            labels_frame = labels_total[f'{frame_num:04}']
            for label in labels_frame:
                bbox = label['bbox']
                bbox = [round(x) for x in bbox]
                im = cv2.rectangle(img=im, pt1=(bbox[0], bbox[1]), pt2=(bbox[2], bbox[3]), color=COLORS[idx], thickness=2)

        frames.append(im)

    if saveFrames:
        path = 'frames_stored'
        os.makedirs(path, exist_ok=True)
        print(f'storing frames with bboxes plotted in it in /{path}...')
        for idx, frame in tqdm(enumerate(frames)):
            cv2.imwrite(f'{path}/{idx}.png', frame)

        print('generating GIF...')
        generate_gifs_from_frames(modelName, initalFrame, finalFrame)

    return frames


def generate_gifs_from_frames(modelName, initialFrame, finalFrame):

    dir = "gifs_stored"
    fp_out = dir + f'/{modelName}.gif'
    os.makedirs(dir, exist_ok=True)

    frames = []
    for idx in range(finalFrame-initialFrame):
        img = Image.open(f'frames_stored/{idx}.png')
        img = img.resize((640, 360))
        frames.append(img)

    frame_one = frames[0]
    frame_one.save(fp=fp_out, format="GIF", append_images=frames, save_all=True, duration=140, loop=0)

    """# filepaths
    fp_in = "frames_stored/*.png"
    dir = "gifs_stored"
    fp_out = dir + f'/{modelName}.gif'

    os.makedirs(dir, exist_ok=True)

    imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
    img = next(imgs)  # extract first image from iterator
    img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=25, loop=0)"""
