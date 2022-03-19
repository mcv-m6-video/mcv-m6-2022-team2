 import scipy.stats as stats
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2

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
