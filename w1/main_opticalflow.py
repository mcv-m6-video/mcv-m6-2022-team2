import os
from OF_utils import *

if __name__ == "__main__":
    # Paths to the corresponding folders
    root = '../'
    gt_path = os.path.join(root, 'data_stereo_flow/training/flow_noc')          # Path of the ground truths
    detections_path = os.path.join(root, 'results_opticalflow_kitti/results')   # Path of the detections
    img_dir = os.path.join(root, 'data_stereo_flow/training/colored_0')         # Path of the images

    gt_OF, det_OF, img = {}, {}, {}         # Dictionaries of sequence_number: Optical flow
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

            # Read the image in which the optical flow is computed
            img.update({seq_number: cv2.cvtColor(cv2.imread(os.path.join(img_dir, f'{seq_number}.png')), cv2.COLOR_BGR2RGB)})

            # Put occluded zones to 0 as only the non-occluded areas have to be computed
            occluded_idx = gt_OF[seq_number][:,:,2] == 0
            det_OF[seq_number][occluded_idx, :] = 0

            # Compute the vector difference between the ground truth and the detection OF
            nocc_error, error = compute_vector_dif(ground_truth=gt_OF[seq_number], detection=det_OF[seq_number])

            print(f'\tmsen: {compute_msen(nocc_error)}')    # Mean Squared Error Of Non-Occluded areas
            print(f'\tpepn: {compute_pepn(nocc_error)}')    # Percentage of Erroneous Pixels in Non-occluded areas

            # Plot the magnitude and direction of the GT and Detected OFs
            draw_OF_magnitude_direction(flow=gt_OF[seq_number])

            plot_OF_errors(error)

            OF_quiver_visualize(img=img[seq_number], flow=gt_OF[seq_number], step=8)




