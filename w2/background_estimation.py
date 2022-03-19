import cv2
from tqdm import tqdm
import numpy as np
import pickle
from os.path import exists
from utils import plot_gaussian_single_pixel
from matplotlib import pyplot as plt
from utils import plotBBox, read_frames
from dataset_gestions import update_labels
import os

def single_gaussian_estimation(frames_paths, alpha=2, plot_results=False):
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
    
    if not os.path.exists('task1_plots'):
        os.makedirs('task1_plots/frames')
        os.makedirs('task1_plots/plot_mean')
        
        for idx, frame in tqdm(enumerate(frames[n_frames_modeling_bg:n_frames_modeling_bg+100])):
            mean_px = mean[646,681]
            std_px = std[646,681]
            mean_px = np.repeat([mean_px],100)
            std_px = np.repeat([std_px],100)
            x = np.arange(n_frames_modeling_bg,n_frames_modeling_bg+100)
            
            frame = cv2.imread(frames_paths[n_frames_modeling_bg+idx])
            frame_aux = frame.copy()
            frame_aux = cv2.rectangle(img=frame_aux, pt1=(626, 661), pt2=(666, 701), color=(0,0,255), thickness=10)
            
            if idx < 10:
                idx_txt = '0' + str(idx)
            else:
                idx_txt = str(idx)
                
            cv2.imwrite("task1_plots/frames/frame_" + idx_txt + '.png',frame_aux)
            
            plt.plot(x,mean_px,color='black', label="Pixel's mean")
            print((mean_px + alpha * (2 + std_px)).shape)
            plt.plot(x,mean_px + alpha * (2 + std_px), linestyle='--',color='blue',label="Detection threshold")
            plt.plot(x,mean_px - alpha * (2 + std_px), linestyle='--',color='blue')
            plt.plot(x[:idx+1],frames[n_frames_modeling_bg:n_frames_modeling_bg+idx+1,646,681],color="red",label="Pixel's value")
            plt.ylim(0,255)
            plt.xlabel("Frame")
            plt.ylabel("Grayscale value")
            if idx == 0:
                plt.legend()
            plt.savefig("task1_plots/plot_mean/frame_" + idx_txt + '.png')

    # Segment foreground and background with the model obtained before
    labels = segment_fg_bg(frames[n_frames_modeling_bg:], n_frames_modeling_bg, mean, std, alpha, plot_results=plot_results)

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


def segment_fg_bg(frames, n_frames_modeling_bg, mean, std, alpha, plot_results=False):
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

        # Morphological operations to filter noise and make a more robust result
        bg_preprocessed = preprocess_mask(bg)

        # Computes bboxes from the mask preprocessed of the frame
        bboxes, stats = extract_bboxes_from_bg(bg_preprocessed)

        # Upload bboxes in dictionary of detections
        for x_top_left, y_top_left, weight, height in bboxes:
            labels = update_labels(labels, idx_frame + n_frames_modeling_bg,
                                             x_top_left, y_top_left, x_top_left + weight, y_top_left + height, 1.)

        if plot_results:
            frame = plotBBox([frame], 0, 1, predicted=stats[:,:4])
            """ plt.imshow(frame[0])
            plt.pause(0.05) """
        
            # TODO, gráfico mostrando la media y alpha*(2+std) de un pixel y su valor a lo largo del tiempo: x=681 y=646

    print('Finished!')

    return labels


def preprocess_mask(bg):
    """
    Morphological operations to filter noise and make a more robust result
    :param bg: mask of the bg and fg
    :return: bg_closed: preprocessed bg and fg
    """

    bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8))
    bg = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, kernel=np.ones((30, 50), np.uint8))
    """ bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, kernel=np.ones((60, 1), np.uint8)) """

    
    return bg


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