import uuid
from dataset_gestions import load_labels
from dataset_gestions import update_labels, write_predictions
import numpy as np
from metric_functions import iou_from_list_of_ground_truths

def tracking_overlap(labels):

    new_track = 0
    initialize = True
    
    for frame, label in labels.items():
        if initialize: # If we are in the first frame
            for detection in label: # Create a new track for each detection
                id = new_track
                new_track += 1
                update_labels(labels, int(frame), id, detection['bbox'][0], detection['bbox'][1],
                              detection['bbox'][2] - detection['bbox'][0], detection['bbox'][3] - detection['bbox'][1], detection['confidence'])
            initialize = False
            
        else: # If we are not in the first frame
            gt_bboxes = np.zeros((len(past_label), 4))
            for idx, past_detections in enumerate(past_label):
                gt_bboxes[idx, :] = np.array((past_detections['bbox'][0], past_detections['bbox'][1],
                                              past_detections['bbox'][0] + past_detections['bbox'][2], 
                                              past_detections['bbox'][1] + past_detections['bbox'][3]))
            for detection in label:
                bbox = [detection['bbox'][0], detection['bbox'][1], 
                        detection['bbox'][2], detection['bbox'][3]]

                ious = iou_from_list_of_ground_truths(gt_bboxes, bbox)

                if max(ious) > 0.5: # If the biggest overlap is higher than a threshold
                    id = past_label[np.argmax(ious)]['id'] # Asign the current bbox to the same track as the overlaping bbox from the previous frame
                    gt_bboxes[np.argmax(ious)] = [-1,-1,-1,-1] # Disable the asigned bbox from the list, not to be assigned again

                else: # If the biggest overlap is samaller than a threshold
                    id = new_track # Initialize a new track
                    new_track += 1

                update_labels(labels, int(frame), id, detection['bbox'][0], detection['bbox'][1],
                              detection['bbox'][2] - detection['bbox'][0], detection['bbox'][3] - detection['bbox'][1], detection['confidence'])
                
                
        past_label = label # Save current tracks to use them in the next iteration


def tracking_kalman():
    print('todo')


if __name__ == "__main__":

    # ........TESTING IF TASK 2.1 WORKS........

    # paths to models
    model_name = 'faster_rcnn_X_101_32x8d_FPN_3x.yaml'
    real_model_name = model_name.replace('.yaml', '')
    path_detections = 'fine_tune'

    # load labels
    detections = load_labels(path_detections, real_model_name + '.txt')

    tracking_overlap(detections)

    path = 'tracking_by_overlap'
    write_predictions(path, detections, real_model_name)

    print('finished')