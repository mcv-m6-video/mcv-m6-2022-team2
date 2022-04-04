import numpy as np
import png
import matplotlib.pyplot as plt
import cv2

def read_OF(OF_path):
    """
    Function to read an Optical Flow png image
    :param OF_path: str, path to the corresponding png file
    :return: np.array
    """

    flow_object = png.Reader(OF_path).asDirect()            # PNG object
    (width, height) = flow_object[3]['size']                # Optical Flow size
    flow_data = list(flow_object[2])                        # Optical Flow data

    flow = np.zeros((height, width, 3), dtype=np.float64)   # Object in which the optical flow is stored

    for i in range(len(flow_data)):
        flow[i, :, 0] = flow_data[i][0::3]
        flow[i, :, 1] = flow_data[i][1::3]
        flow[i, :, 2] = flow_data[i][2::3]

    invalid_idx = (flow[:,:,2] == 0)
    flow[:,:,0:2] = (flow[:,:,0:2] - 2 ** 15) / 64.0
    flow[invalid_idx, 0] = 0
    flow[invalid_idx, 1] = 1

    return flow

def compute_vector_dif(ground_truth, detection):
    """
    Compute the difference between two Optical Flows
    :param ground_truth: np.array
    :param detection: np.array
    :return:
    np.array, vector difference between 2 OFs without counting the occluded areas
    np.array, vector difference between 2 OFs
    """

    distance = detection[:,:,:2] - ground_truth[:,:,:2]     # Distance vector
    error = np.sqrt(np.sum(distance ** 2, axis=2))          # Squared error

    # discard vectors which from occluded areas (occluded = 0)
    non_occluded_idx = ground_truth[:, :, 2] != 0

    return error[non_occluded_idx], error

def compute_msen(error):
    """Mean of the squared error"""
    return np.mean(error)

def compute_pepn(error, th=3):
    """Percentage of Erroneous Pixels"""
    return np.sum(error>th) / len(error)

def draw_OF_magnitude_direction(flow):
    """
    Function to plot the magnitude and direction of an OF
    :param flow: np.array
    """
    # Compute the magnitude of the ground truth OF
    flow_magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

    # Clip the highest magnitude values according to the 0.95 quantile
    clip_th = np.quantile(flow_magnitude, 0.98)
    flow_magnitude = np.clip(flow_magnitude, 0, clip_th)

    # Normalize
    flow_magnitude = (flow_magnitude / np.max(flow_magnitude)) * 255

    # Compute the direction of the ground truth OF
    flow_direction = np.arctan2(flow[:, :, 1], flow[:, :, 0]) + np.pi
    flow_hsv = np.zeros(flow.shape, dtype=np.uint8)
    flow_hsv[:, :, 0] = flow_direction / (2 * np.pi) * 179
    flow_hsv[:, :, 1] = flow_magnitude
    flow_hsv[:, :, 2] = 255
    flow_direction = cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2RGB)

    # Plots
    plt.subplot(211)
    plt.title('OF magnitude')
    plt.imshow(flow_magnitude, cmap='gray')
    plt.subplot(212)
    plt.title('OF direction')
    plt.imshow(flow_direction, cmap='viridis')
    plt.show()

def plot_OF_errors(error):
    """
    Plot flow errors and error histograms
    :param error: np.array, vector error
    """

    mu = np.mean(error)
    std = np.std(error)

    plt.subplot(211)
    plt.imshow(error, cmap='afmhot')

    plt.subplot(212)
    error = error[error!=1]
    plt.hist(error.ravel(), bins=100)
    xmin, xmax = plt.xlim()
    _, ymax = plt.ylim()
    x = np.linspace(xmin, xmax, 100)
    y = ymax * np.exp(-(x-mu) ** 2 / (2 * std **2))
    plt.plot(x, y)
    plt.text(xmax - 0.2*xmax, ymax - 0.2*ymax, f"Mean: {mu:.2f}\nStd: {std:.2f}", size=10, bbox=dict(boxstyle='round',
                                                                                                     ec='darkorange',
                                                                                                     fc='orange'))
    plt.show()

def OF_quiver_visualize(img, flow, step):
    """
    Plot the OF through quiver function
    :param img: the scene RGB image
    :param flow: the Optical flow image (GT or Estimated)
    :param step: Step controls the sampling to draw the arrows
    """
    # The OF is composed by a 2D object indicating the movement vector of each pixel (x and y-axis). A third component
    # is added to indicate if the pixel is occluded in the other image or not.
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    occ = flow[:, :, 2]

    # If occlusion, vector set to 0
    U = u * occ
    V = v * occ

    (h, w) = flow.shape[0:2]

    M = np.hypot(u, v)  # color

    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))  # initial

    # Plot figure
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.quiver(x[::step, ::step], y[::step, ::step], U[::step, ::step], V[::step, ::step],
               M[::step, ::step], scale_units='xy', angles='xy', color=(1,0,0,1))
    plt.axis('off')
    plt.show()