import scipy.stats as stats
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
from os.path import exists
import pickle
from tqdm import tqdm
import os

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
        for idx, (name, labels_total) in enumerate(labels.items()):
            labels_total
            for bbox in labels_total:
                bbox = [round(x) for x in bbox]
                im = cv2.rectangle(img=im, pt1=(bbox[0], bbox[1]), pt2=(bbox[0] + bbox[2], bbox[1] + bbox[3]), color=COLORS[idx], thickness=2)

        frames.append(im)

    return frames


def read_frames(frames_paths,color=False):
    """
    Reads all frames and store them in a pickle in order to be more efficient
    :param frames_paths:
    :return: frames: variable with all the frames stacked
    """
    
    if not color and exists('variables/frames.pickle'):
        with open('variables/frames.pickle', 'rb') as f:
            frames = pickle.load(f)
    elif color and exists('variables/frames_channel0.pickle'):
        return
    else:
        frames = []
        print('reading frames and storing them in variables/frames.pickle in order to be more efficient...')
        for f_path in tqdm(frames_paths):
            frame = cv2.imread(f_path)
            if not color:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)

        frames = np.array(frames)
        if not color:
            with open('variables/frames.pickle', 'wb') as f:
                pickle.dump(frames, f)
        else:
            for c in range(3):
                print(c)
                with open(f'variables/frames_channel{c}.pickle', 'wb') as f:
                    pickle.dump(frames[:,:,:,c], f)
            return

    return frames

def plot_pixel_detection(frames,mean,std,alpha,n_frames):
    if not exists('task2_plots'):
        os.makedirs('task2_plots/frames')
        os.makedirs('task2_plots/plot_mean')
        
        for idx, frame in tqdm(enumerate(frames)):
            mean_px = mean[:idx+1,646,681]
            std_px = std[:idx+1,646,681]
            x = np.arange(n_frames,n_frames+100)
            
            frame_aux = frame.copy()
            frame_aux = cv2.cvtColor(frame_aux, cv2.COLOR_GRAY2BGR)

            frame_aux = cv2.rectangle(img=frame_aux, pt1=(626, 661), pt2=(666, 701), color=(0,0,255), thickness=10)
            
            if idx < 10:
                idx_txt = '0' + str(idx)
            else:
                idx_txt = str(idx)
                
            cv2.imwrite("task2_plots/frames/frame_" + idx_txt + '.png',frame_aux)
            
            plt.plot(x[:idx+1],mean_px,color='black', label="Pixel's mean")
            plt.plot(x[:idx+1],mean_px + alpha * (2 + std_px), linestyle='--',color='blue',label="Detection threshold")
            plt.plot(x[:idx+1],mean_px - alpha * (2 + std_px), linestyle='--',color='blue')
            plt.plot(x[:idx+1],frames[:idx+1,646,681],color="red",label="Pixel's value")
            plt.ylim(0,255)
            plt.xlim(x[0],x[-1])
            plt.xlabel("Frame")
            plt.ylabel("Grayscale value")
            if idx == 0:
                plt.legend()
            plt.savefig("task2_plots/plot_mean/frame_" + idx_txt + '.png')
        
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

