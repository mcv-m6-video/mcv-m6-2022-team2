import cv2
from tqdm import tqdm
import numpy as np
import pickle
from os.path import exists
from utils import plot_gaussian_single_pixel
from matplotlib import pyplot as plt
from utils import plotBBox, plot_pixel_detection
from dataset_gestions import update_labels
import os


def single_gaussian_estimation(frames, alpha=0.15, rho=0, adaptive=False, plot_results=False):
    """
    It is the MOTHER FUNCTION. It estimates the background and foreground, computes the bounding boxes
    from the masks with several techniques and returns the labels dictionary of lists updated.
    :param frames_paths: paths where are located all the frames
    :param alpha: parameter to threshold the gaussian of each frame and be more permisive or not
    :param plot_results: true to plot some results
    :return: labels: labels dictionary of lists updated, where are all the bboxes, confidence, etc.
    """

    n_frames = round(frames.shape[0] * 0.25) # First 25% of the frame sequence to model background
    training_frames = frames[:n_frames]
    segmenting_frames = frames[n_frames:]

    mean, std = mean_std(training_frames)   # Model the background as a single gaussian (mean, std)

    # Segment foreground and background with the model obtained before
    labels = segmentation(segmenting_frames, n_frames, mean, std, alpha, rho,
                           adaptive=adaptive, plot_results=plot_results)

    if plot_results:
        plot_frames = frames[n_frames:n_frames+100]
        plot_pixel_detection(plot_frames,mean,std,alpha,n_frames)

    return labels


def mean_std(frames):
    """
    Estimates the mean and std matrix from the 25% of the frames from the video. If it doesn't exist the
    variables, it creates them, if they exist, loads them to be more efficient and less time consuming.
    :param frames_paths: paths of the frames
    :return: mean and std matrices from all the frames (of size of the image)
    """

    if exists('variables/mean_single_gauss.pickle') and exists('variables/std_single_gauss.pickle'):
        print('Loading stored variables (mean and std from the image)...')
        with open('variables/mean_single_gauss.pickle', 'rb') as f:
            mean = pickle.load(f)
        with open('variables/std_single_gauss.pickle', 'rb') as f:
            std = pickle.load(f)

    else:
        print('Modeling bg as a single gaussian ...')

        mean = np.mean(frames, axis=0)
        std = np.std(frames, axis=0)

        print('Storing mean and std...')
        with open('variables/mean_single_gauss.pickle', 'wb') as f:
            pickle.dump(mean, f)
        with open('variables/std_single_gauss.pickle', 'wb') as f:
            pickle.dump(std, f)

    return mean, std

def adaptive_mean_std(frame, mask, rho, mean, std):

    [pos_x, pos_y] = np.where(mask == 0)  # If the pixel belongs to the background...

    # Recompute mean and std of the background pixels
    mean[pos_x, pos_y] = rho * frame[pos_x, pos_y] + (1 - rho) * mean[pos_x, pos_y]
    std[pos_x, pos_y] = np.sqrt(rho * (frame[pos_x, pos_y] - mean[pos_x, pos_y])**2 + (1 - rho) * std[pos_x, pos_y]**2)

    return mean, std


def background_mask(frame, mean, std, alpha):
    """
    Generate the background mask for a frame
    :param frame: img frame
    :param mean: mean img
    :param std: std image
    :param alpha: parameter to threshold the detection
    :return: mask: 0 if bg, 255 if fg
    """

    # Background is the same shape as frame
    mask = np.zeros_like(frame)

    # Perform background and foreground substraction
    diff = frame - mean
    foreground_idx = np.where(abs(diff) >= alpha * (2 + std))

    # If pixel is classified as foreground, assign a 255 to it. If not, pixel is bg and remains at 0.
    mask[foreground_idx[0], foreground_idx[1]] = 255
    


    return mask

def preprocess_mask(bg):
    """
    Morphological operations to filter noise and get a more robust result
    :param bg: mask of the bg and fg
    :return: bg_closed: preprocessed bg and fg
    """
    
    bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8))
    bg = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, kernel=np.ones((50, 50), np.uint8))

    return bg


def segmentation(frames, n_frames, mean, std, alpha, rho, adaptive=False, plot_results=True):
    """
    Compute background and foreground for every frames.
    :param frames: Variable where all the frames are stacked
    :param n_frames_modeling_bg: number of the frames of the first 25% of the test sequence to model background
    :param mean: Mean image of the first 25% frames
    :param std: Std image of the first 25% frames
    :param alpha: Parameter to control thresholding on the gaussian estimation
    :return: labels: dictionary of all the detections with bboxes, confidence=1, etc. as in week1
    """

    labels = {}

    print('Segmenting foreground and background...')
    for idx_frame, frame in enumerate(tqdm(frames)):

        mask = background_mask(frame, mean, std, alpha)   # Compute the mask of the frame

        if adaptive:
            mean, std = adaptive_mean_std(frame, mask, rho, mean, std)
            mask = background_mask(frame, mean, std, alpha)   # Recompute the mask of the frame

        mask = preprocess_mask(mask) # Morphological operations to filter noise and make a more robust result
        bboxes = foreground_bboxes(mask) # Computes bboxes from the mask

        # Upload bboxes in dictionary of detections
        for x, y, weight, height in bboxes:
            labels = update_labels(labels, idx_frame + n_frames,
                                   x, y, x + weight, y + height, 1.)

        if plot_results:
            frame = plotBBox([frame], 0, 1, predicted=bboxes)
            plt.imshow(frame[0])
            plt.pause(0.05)

    plt.show()

    print('Finished!')

    return labels

def foreground_bboxes(bg_preprocessed):
    """
    Computes bboxes from the mask preprocessed of the frame
    :param bg_preprocessed: mask preprocessed
    :return: list of bboxes
    """

    # Compute connected components. Stats: [xL, yL, w, h, area]
    _, _, stats, _ = cv2.connectedComponentsWithStats(bg_preprocessed)

    # Sorted by area: The largest, at the end.
    stats = stats[stats[:, 4].argsort()]
    # todo: cambiar de orden las bboxes y organizarlas al reves?

    # Save all the bboxes from that frame.
    bboxes = []
    for stat in stats: # First connected components corresponds to the background
        if stat[3] > 50 and stat[3] < 500 and stat[2] > 50 and stat[2] < 500:  # Filter minimum area
            bboxes.append([stat[0], stat[1], stat[2], stat[3]])  # x, y, w, h

    return bboxes