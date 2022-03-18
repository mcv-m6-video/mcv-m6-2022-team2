import cv2
from tqdm import tqdm
import numpy as np
import pickle
from os.path import exists
from utils import plot_gaussian_single_pixel

def single_gaussian_estimation(frames_paths, alpha=2, plot_results=False):

    # First 25% of the test sequence to model background
    n_frames_modeling_bg = round(len(frames_paths) * 0.25)

    mean, std = model_bg_single_gaussian(frames_paths[:n_frames_modeling_bg])

    if plot_results:
        plot_gaussian_single_pixel(mean, std, pixel=[256, 234])

    #stacked_masks = substract_bg_single_gaussian(mean, std, alpha, frames_paths[n_frames_modeling_bg:])


def model_bg_single_gaussian(frames_paths):
    """
    Estimates the mean and std matrix from the 25% of the frames from the video. If it doesn't exist the
    variables, it creates them, if they exist, loads them to be more efficient and less time consuming.
    :param frames_paths: paths of the frames
    :return: mean and std matrices from all the frames (of size of the image)
    """

    if exists('variables/mean_single_gauss.pickle') and exists('variables/std_single_gauss.pickle'):
        print('loading stored variables')
        with open('variables/mean_single_gauss.pickle', 'rb') as f:
            mean = pickle.load(f)
        with open('variables/std_single_gauss.pickle', 'rb') as f:
            std = pickle.load(f)

    else:
        print('Modeling bg as a single gaussian ...')
        for idx, frame_path in tqdm(enumerate(frames_paths)):
            frame = cv2.imread(frame_path)
            frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if idx == 0:
                stacked_frames = np.empty([frame_bw.shape[0], frame_bw.shape[1], len(frames_paths)])
            stacked_frames[:, :, idx] = frame_bw

        mean = np.mean(stacked_frames, axis=-1)
        std = np.std(stacked_frames, axis=-1)

        print('Storing mean and std...')
        with open('variables/mean_single_gauss.pickle', 'wb') as f:
            pickle.dump(mean, f)
        with open('variables/std_single_gauss.pickle', 'wb') as f:
            pickle.dump(mean, f)

    return mean, std


def bg_single_gaussian_frame(frame_path, mean, std, alpha):

    # todo: Faltan bboxes y preprocesado!

    frame = cv2.imread(frame_path)
    frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bg = np.zeros_like(frame_bw)

    diff = frame_bw - mean
    foreground_idx = np.where(abs(diff) > alpha * (2 + std))
    bg[foreground_idx[0], foreground_idx[1]] = 255

    return bg

# todo:
#  funcion que llame a bg_single_gaussian_frame y calcule masks y bboxes para todos