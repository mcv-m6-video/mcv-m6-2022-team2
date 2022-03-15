import os
import png
import numpy as np
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

def draw_OF_magnitude_direction(gt_flow, det_flow):
    """
    Function to plot the magnitude and direction of 2 OFs
    :param gt_flow: np.array
    :param det_flow: np.array
    """

    # Compute the magnitude of the ground truth OF
    gt_flow_magnitude = np.sqrt(gt_flow[:, :, 0] ** 2 + gt_flow[:, :, 1] ** 2)
    gt_flow_magnitude = (gt_flow_magnitude / np.max(gt_flow_magnitude)) * 255

    # Compute the direction of the ground truth OF
    gt_flow_direction = np.arctan2(gt_flow[:, :, 1], gt_flow[:, :, 0]) + np.pi

    # Compute the magnitude of the detected OF
    det_flow_magnitude = np.sqrt(det_flow[:, :, 0] ** 2 + det_flow[:, :, 1] ** 2)
    det_flow_magnitude = (det_flow_magnitude / np.max(det_flow_magnitude)) * 255

    # Compute the direction of the detected OF
    det_flow_direction = np.arctan2(det_flow[:, :, 1], det_flow[:, :, 0]) + np.pi

    # Plots
    plt.subplot(221)
    plt.title('Magnitude of the Ground Truth Optical Flow')
    plt.imshow(gt_flow_magnitude, cmap='gray')
    plt.subplot(222)
    plt.title('Direction of the Ground Truth Optical Flow')
    plt.imshow(gt_flow_direction, cmap='viridis')
    plt.subplot(223)
    plt.title('Magnitude of the Detected Optical Flow')
    plt.imshow(det_flow_magnitude, cmap='gray')
    plt.subplot(224)
    plt.title('Direction of the Detected Optical Flow')
    plt.imshow(det_flow_direction, cmap='viridis')
    plt.show()

def OF_quiver_visualize(img, flow, step, fname_output='flow_quiver.png'):
    """
    Plot the OF through quiver function
    :param img: the scene RGB image
    :param flow: the Optical flow image (GT or Estimated)
    :param step: Step controls the sampling to draw the arrows
    :param fname_output: name given to the output image to be saved
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    occ = flow[:, :, 2]

    U = u * occ
    V = v * occ

    (h, w) = flow.shape[0:2]

    M = np.hypot(u, v)  # color

    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))  # initial

    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.quiver(x[::step, ::step], y[::step, ::step], U[::step, ::step], V[::step, ::step],
               M[::step, ::step], scale_units='xy', angles='xy', scale=.05, color=(1, 0, 0, 1))
    plt.axis('off')
    plt.show()

def plot_OF_errors(error):

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
    plt.show()

if __name__ == "__main__":
    # Paths to the corresponding folders
    root = '../'
    gt_path = os.path.join(root, 'data_stereo_flow/training/flow_noc')          # Path of the ground truths
    detections_path = os.path.join(root, 'results_opticalflow_kitti/results')   # Path of the detections
    img_dir = os.path.join(root, 'data_stereo_flow/training/colored_0')         # Path of the images

    gt_OF, det_OF = {}, {}                  # Dictionaries of sequence_number: Optical flow
    dif_OF, MSEN_OF, PEPN_OF = {}, {}, {}   # Dictionaries of the metrics

    # Search the sequences in which the OF has been computed
    for det in os.listdir(detections_path):                 # Iterate thorugh all the detection files
        if det[-3:] == 'png':                               # Look only the png files
            seq_number = det[:-4].replace("LKflow_", "")    # Obtain the sequence number

            print(seq_number)                               # Print the sequence number

            # Read the Ground Truth Optical Flow
            gt_OF.update({seq_number: read_OF(os.path.join(gt_path, f'{seq_number}.png'))})

            # Read the detected Optical Flow
            det_OF.update({seq_number: read_OF(os.path.join(detections_path, det))})

            # Put occluded zones to 0 as only the non-occluded areas have to be computed
            occluded_idx = gt_OF[seq_number][:,:,2] == 0
            det_OF[seq_number][occluded_idx, :] = 0

            # Compute the vector difference between the ground truth and the detection OF
            nocc_error, error = compute_vector_dif(ground_truth=gt_OF[seq_number], detection=det_OF[seq_number])

            print(f'\tmsen: {compute_msen(nocc_error)}')    # Mean Squared Error Of Non-Occluded areas
            print(f'\tpepn: {compute_pepn(nocc_error)}')    # Percentage of Erroneous Pixels in Non-occluded areas

            # Plot the magnitude and direction of the GT and Detected OFs
            #draw_OF_magnitude_direction(gt_OF[seq_number], det_OF[seq_number])

            #plot_OF_errors(error)

            OF_quiver_visualize()




