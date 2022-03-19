import cv2
from tqdm import tqdm
import numpy as np
import pickle
from os.path import exists
from utils import plot_gaussian_single_pixel
from matplotlib import pyplot as plt
from utils import plotBBox

def single_gaussian_estimation(frames_paths, alpha=2, plot_results=False):

    if exists('frames.pickle'):
        with open('frames.pickle', 'rb') as f:
            frames = pickle.load(f)
    else:
        frames = []
        for f_path in tqdm(frames_paths):
            frame = cv2.imread(f_path)
            frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame_bw)
            
        frames = np.array(frames)
        with open('frames.pickle', 'wb') as f:
            pickle.dump(frames, f)
            
    print(frames.shape)

    # First 25% of the test sequence to model background
    n_frames_modeling_bg = round(len(frames_paths) * 0.25)

    mean, std = model_bg_single_gaussian(frames[:n_frames_modeling_bg])
    segment_fg_bg(frames[n_frames_modeling_bg:],mean,std,0.3)

    if plot_results:
        plot_gaussian_single_pixel(mean, std, pixel=[256, 234])

    #stacked_masks = substract_bg_single_gaussian(mean, std, alpha, frames_paths[n_frames_modeling_bg:])


def model_bg_single_gaussian(frames):
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

        mean = np.mean(frames, axis=0)
        std = np.std(frames, axis=0)

        print('Storing mean and std...')
        with open('variables/mean_single_gauss.pickle', 'wb') as f:
            pickle.dump(mean, f)
        with open('variables/std_single_gauss.pickle', 'wb') as f:
            pickle.dump(std, f)

    return mean, std


def bg_single_gaussian_frame(frame, mean, std, alpha):

    # todo: Faltan bboxes y preprocesado!

    bg = np.zeros_like(frame)

    diff = frame - mean
    foreground_idx = np.where(abs(diff) > alpha * (2 + std))
    bg[foreground_idx[0], foreground_idx[1]] = 255

    return bg

# todo:
#  funcion que llame a bg_single_gaussian_frame y calcule masks y bboxes para todos

def segment_fg_bg(frames,mean,std,alpha):
    for frame in frames:
        bg = bg_single_gaussian_frame(frame, mean, std, alpha)
        bg_opened = cv2.morphologyEx(bg, cv2.MORPH_OPEN,kernel=np.ones((5,5),np.uint8))
        bg_closed = cv2.morphologyEx(bg_opened, cv2.MORPH_CLOSE,kernel=np.ones((30,30),np.uint8))
        (_, components, stats, _) = cv2.connectedComponentsWithStats(bg_closed)
       
        frame = plotBBox([frame], 0, 1, predicted=stats[:,:4]) # TODO, filtrar connected components por tamaño

        plt.imshow(frame[0])
        plt.pause(0.05)
    
    plt.show()
    
    #TODO, gráfico mostrando la media y alpha*(2+std) de un pixel y su valor a lo largo del tiempo
    for frame in frames:
        plt.plot()
    x=681
    y=646