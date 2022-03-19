import cv2
from tqdm import tqdm
import numpy as np
import pickle
from os.path import exists
from utils import plot_gaussian_single_pixel
from matplotlib import pyplot as plt
from utils import plotBBox, read_frames, plot_pixel_detection
from dataset_gestions import update_labels
import os


def single_gaussian_estimation(frames_paths, alpha=0.15, rho=0, make_estimation_adaptive=False, plot_results=False):
    """
    It is the MOTHER FUNCTION. It estimates the background and foreground, computes the bounding boxes
    from the masks with several techniques and returns the labels dictionary of lists updated.
    :param frames_paths: paths where are located all the frames
    :param alpha: parameter to threshold the gaussian of each frame and be more permisive or not
    :param plot_results: true to plot some results
    :return: labels: labels dictionary of lists updated, where are all the bboxes, confidence, etc.
    """

    # Read all frames from the paths
    frames = read_frames(frames_paths)

    # First 25% of the test sequence to model background
    n_frames_modeling_bg = round(len(frames_paths) * 0.25)

    # Model the bg as a single gaussian
    mean, std = model_bg_single_gaussian(frames[:n_frames_modeling_bg])

    if plot_results and not exists('task1_plots'):
        plot_pixel_detection(frames[n_frames_modeling_bg:n_frames_modeling_bg+100],mean,std,alpha,n_frames_modeling_bg)

    # Segment foreground and background with the model obtained before
    labels = segment_fg_bg(frames[n_frames_modeling_bg:], n_frames_modeling_bg, mean, std, alpha, rho,
                           make_estimation_adaptive=make_estimation_adaptive, plot_results=plot_results)

    # If plot results is true, plot graphics
    # if plot_results:
    #    plot_gaussian_single_pixel(mean, std, pixel=[256, 234])

    return labels


def model_bg_single_gaussian(frames):
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


def bg_single_gaussian_frame(frame, mean, std, alpha):
    """
    Generate the bg and fg mask for a single frame
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
    foreground_idx = np.where(abs(diff) > alpha * (2 + std))

    # If pixel is classified as foreground, assign a 255 to it. If not, pixel is bg and remains at 0.
    mask[foreground_idx[0], foreground_idx[1]] = 255

    return mask


def segment_fg_bg(frames, n_frames_modeling_bg, mean, std, alpha, rho, make_estimation_adaptive=False,
                  plot_results=True):
    """
    Compute background and foreground for all frames.
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

        # Compute mask for each frame
        bg = bg_single_gaussian_frame(frame, mean, std, alpha)

        if make_estimation_adaptive:
            mean, std = change_gaussian_parameters(frame, bg, rho, mean, std)

        # Morphological operations to filter noise and make a more robust result
        bg_preprocessed = preprocess_mask(bg)

        """plt.imshow(bg_preprocessed)
        plt.pause(0.01)"""

        # Computes bboxes from the mask preprocessed of the frame
        bboxes, stats = extract_bboxes_from_bg(bg_preprocessed)

        # Upload bboxes in dictionary of detections
        for x_top_left, y_top_left, weight, height in bboxes:
            labels = update_labels(labels, idx_frame + n_frames_modeling_bg,
                                   x_top_left, y_top_left, x_top_left + weight, y_top_left + height, 1.)

        if plot_results:
            frame = plotBBox([frame], 0, 1, predicted=stats[:, :4])
            plt.imshow(frame[0])
            plt.pause(0.05)

    plt.show()

    print('Finished!')

    return labels


def preprocess_mask(bg):
    """
    Morphological operations to filter noise and make a more robust result
    :param bg: mask of the bg and fg
    :return: bg_closed: preprocessed bg and fg
    """

    bg_opened = cv2.morphologyEx(bg, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8))
    bg_closed = cv2.morphologyEx(bg_opened, cv2.MORPH_CLOSE, kernel=np.ones((30, 30), np.uint8))

    return bg_closed


def extract_bboxes_from_bg(bg_preprocessed):
    """
    Computes bboxes from the mask preprocessed of the frame
    :param bg_preprocessed: mask preprocessed
    :return: list of bboxes
    """

    # Compute connected components. Stats: [xL, yL, w, h, area]
    (_, components, stats, _) = cv2.connectedComponentsWithStats(bg_preprocessed)

    # Sorted by area: The largest, at the end.
    stats = stats[stats[:, 4].argsort()]
    # todo: cambiar de orden las bboxes y organizarlas al reves?

    # Save all the bboxes from that frame.
    bboxes = []
    for stat in stats:
        # Filter for area, if below that region, we drop the bbox
        if stat[4] > 1500:
            bboxes.append([stat[0], stat[1], stat[2], stat[3]])  # x, y, h, w

    # Drop last bbox, it represents all the image.
    bboxes = bboxes[:-1]

    return bboxes, stats


def change_gaussian_parameters(frame, mask, rho, mean, std):

    # If the pixel belongs to the background...
    [pos_x, pos_y] = np.where(mask == 0)

    # recompute mean and std in all the positions where the pixel belongs to the background
    mean[pos_x, pos_y] = rho * frame[pos_x, pos_y] + (1 - rho) * mean[pos_x, pos_y]
    std[pos_x, pos_y] = rho * np.square(frame[pos_x, pos_y] - mean[pos_x, pos_y]) + (1 - rho) * std[pos_x, pos_y]

    return mean, std