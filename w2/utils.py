import scipy.stats as stats
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
from os.path import exists
import pickle
import tqdm

def plot_gaussian_single_pixel(mean, std, pixel):
    """
    Draws the estimated gaussian in a determined pixel.
    :param mean: mean matrix
    :param std: std matrix
    :param pixel: [coordinate1, coordinate2]
    """
    mu = mean[pixel[0], pixel[1]]
    variance = std[pixel[0], pixel[1]]
    print(f'for pixel: {pixel}, the mean and variance is:')
    print(f'mean: {mu}')
    print(f'variance: {variance}')
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.title(f'modeling of pixel {pixel}')
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.show()
    
def plotBBox(imgs, initalFrame, finalFrame, **labels):
    """
    plots bboxes in a frame
    todo: comentar igor (no he hecho yo la funcion)
    :param imgs:
    :param initalFrame:
    :param finalFrame:
    :param labels:
    :return:
    """
    frames = []
    COLORS=[(0,255,0), (0,0,255)]
    for frame_num in range(initalFrame, finalFrame):
        im = imgs[frame_num]
        print(labels)
        for idx, (name, labels_total) in enumerate(labels.items()):
            labels_total
            for bbox in labels_total:
                bbox = [round(x) for x in bbox]
                im = cv2.rectangle(img=im, pt1=(bbox[0], bbox[1]), pt2=(bbox[0] + bbox[2], bbox[1] + bbox[3]), color=COLORS[idx], thickness=2)

        frames.append(im)

    return frames


def read_frames(frames_paths):
    """
    Reads all frames and store them in a pickle in order to be more efficient
    :param frames_paths:
    :return: frames: variable with all the frames stacked
    """
    if exists('variables/frames.pickle'):
        with open('variables/frames.pickle', 'rb') as f:
            frames = pickle.load(f)
    else:
        frames = []
        print('reading frames and storing them in variables/frames.pickle in order to be more efficient...')
        for f_path in frames_paths:
            frame = cv2.imread(f_path)
            frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame_bw)

        frames = np.array(frames)
        with open('variables/frames.pickle', 'wb') as f:
            pickle.dump(frames, f)

    return frames

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

