import cv2
from tqdm import tqdm
import numpy as np
import pickle
from os.path import exists
from utils import plot_gaussian_single_pixel
from matplotlib import pyplot as plt
from utils import plotBBox, read_frames

def single_gaussian_estimation(frames_paths, alpha=2, plot_results=False):

    # Read all frames from the paths
    frames = read_frames(frames_paths)

    # First 25% of the test sequence to model background
    n_frames_modeling_bg = round(len(frames_paths) * 0.25)

    # Model the bg as a single gaussian
    mean, std = model_bg_single_gaussian(frames[:n_frames_modeling_bg])

    # Segment foreground and background with the model obtained before
    segment_fg_bg(frames[n_frames_modeling_bg:], mean, std, alpha=0.3)

    # If plot results is true, plot graphics
    if plot_results:
        plot_gaussian_single_pixel(mean, std, pixel=[256, 234])


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

    # Background is the same shape as frame
    mask = np.zeros_like(frame)

    # Perform background and foreground substraction
    diff = frame - mean
    foreground_idx = np.where(abs(diff) > alpha * (2 + std))

    # If pixel is classified as foreground, assign a 255 to it. If not, pixel is bg and remains at 0.
    mask[foreground_idx[0], foreground_idx[1]] = 255

    return mask


def segment_fg_bg(frames, mean, std, alpha, plot_results=False):
    """
    Compute background and foreground for all frames.
    :param frames: Variable where all the frames are stacked
    :param mean: Mean image of the first 25% frames
    :param std: Std image of the first 25% frames
    :param alpha: Parameter to control thresholding on the gaussian estimation
    :return: todo: ?????
    """
    print('Segmenting foreground and background...')
    for frame in tqdm(frames):
        # Compute mask for each frame
        bg = bg_single_gaussian_frame(frame, mean, std, alpha)

        # Morphological operations to filter noise and make a more robust result
        bg_preprocessed = preprocess_mask(bg)

        # Computes bboxes from the mask preprocessed of the frame
        bboxes, stats = extract_bboxes_from_bg(bg_preprocessed)

        if plot_results:
            frame = plotBBox([frame], 0, 1, predicted=stats[:,:4])
            plt.imshow(frame[0])
            plt.pause(0.05)
            # TODO, gráfico mostrando la media y alpha*(2+std) de un pixel y su valor a lo largo del tiempo: x=681 y=646


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

    # Save all the bboxes from that frame.
    bboxes = []
    for stat in stats:
        # Filter for area, if below that region, we drop the bbox
        if stat[4] > 50:
            bboxes.append([stat[0], stat[1], stat[2], stat[3]])  # x, y, h, w

    # todo: Drop last bbox, it represents all the image. No se si hará petar el codigo para cuando no hay...

    return bboxes, stats