import scipy.stats as stats
import math
import matplotlib.pyplot as plt
import numpy as np

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
