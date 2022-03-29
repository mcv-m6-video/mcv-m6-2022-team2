import uuid
from dataset_gestions import load_labels
from dataset_gestions import update_labels, write_predictions
import numpy as np
from metric_functions import iou_from_list_of_ground_truths
import motmetrics as mm


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
            past_bboxes = np.zeros((len(past_label), 4))
            for idx, past_detections in enumerate(past_label): # Take all previous frame detections
                past_bboxes[idx, :] = np.array((past_detections['bbox'][0], past_detections['bbox'][1],
                                              past_detections['bbox'][0] + past_detections['bbox'][2], 
                                              past_detections['bbox'][1] + past_detections['bbox'][3]))
               
            for detection in label: # Compare each current detection with all the previous detections
                bbox = [detection['bbox'][0], detection['bbox'][1], 
                        detection['bbox'][2], detection['bbox'][3]]
                
                ious = iou_from_list_of_ground_truths(past_bboxes, bbox)

                if np.nanmax(ious) > 0.5: # If the biggest overlap is higher than a threshold
                    id = past_label[np.nanargmax(ious)]['id'] # Asign the current bbox to the same track as the overlaping bbox from the previous frame
                    past_bboxes[np.nanargmax(ious)] = [np.nan,np.nan,np.nan,np.nan] # Disable the assigned bbox from the list, not to be assigned again

                else: # If the biggest overlap is samaller than a threshold
                    id = new_track # Initialize a new track
                    new_track += 1

                update_labels(labels, int(frame), id, detection['bbox'][0], detection['bbox'][1],
                              detection['bbox'][2] - detection['bbox'][0], detection['bbox'][3] - detection['bbox'][1], detection['confidence'])
                
                
        past_label = label # Save current tracks to use them in the next iteration


def tracking_kalman():
    print('todo')
    
def metrics(ground_truth,detections): # Task 2.
    acc = mm.MOTAccumulator(auto_id=True)
    
    for key in detections.keys():
        gt_bboxes = np.zeros((len(ground_truth[key]), 4))
        
        gt_index = []
        for idx, gt in enumerate(ground_truth[key]):
            gt_bboxes[idx,:] = gt['bbox']
            gt_index.append(int(gt['id']))         
            
            
        det_index = []
        ious_per_detection = []
        for idx,det in enumerate(detections[key]):
            bbox = [det['bbox'][0], det['bbox'][1], 
                    det['bbox'][0] + det['bbox'][2], 
                    det['bbox'][1] + det['bbox'][3]]

            ious = iou_from_list_of_ground_truths(gt_bboxes, bbox)
            ious = np.where(ious > 0.5, ious, np.nan)
            
            det_index.append(det['id'])
            
            ious_per_detection.append(ious)

        ious_per_detection = np.array(ious_per_detection).T.tolist()
        acc.update(
            gt_index,                     # Ground truth objects in this frame
            det_index,             # Detector hypotheses in this frame
            ious_per_detection
        )
        
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['num_frames', 'idp', 'idr','idf1','recall','precision'], name='acc')
    print(summary)


if __name__ == "__main__":

    # ........TESTING IF TASK 2.1 WORKS........
    # Directions where all the sequence is located
    path_data = '../../data/AICity_data/train/S03/c010'

    # Direction where all the ground truth annotations are located
    path_gt = path_data + '/gt'

    # Direction where the video of the sequence is located
    path_video = path_data + '/vdo'

    # Load and update ground truth labels
    ground_truth = load_labels(path_gt, 'w1_annotations.xml')  # ground_truth = load_labels(path_gt, 'gt.txt')

    # paths to models
    #model_name = 'retinanet_R_101_FPN_3x.yaml'
    model_name = 'faster_rcnn_X_101_32x8d_FPN_3x.yaml'
    real_model_name = model_name.replace('.yaml', '')
    path_detections = 'fine_tune'

    # load labels
    detections = load_labels(path_detections, real_model_name + '.txt')
    
    
    ground_truth_list = list(ground_truth.keys())
    ground_truth_list.sort()
    print(ground_truth['0001'])
    print(list(detections.keys())[0])

    tracking_overlap(detections)
    
        
    path = 'tracking_by_overlap'
    write_predictions(path, detections, real_model_name)
    
    metrics(ground_truth,detections) # Task 2.3

    print('finished')