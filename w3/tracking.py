import uuid
from dataset_gestions import load_labels, get_frames_paths
from dataset_gestions import update_labels, write_predictions
import numpy as np
from metric_functions import iou_from_list_of_ground_truths
import motmetrics as mm
import cv2
from PIL import Image


def tracking_overlap(labels,frames_path,save=False, gif=False):

    new_track = 0
    initialize = True
    done = True
    
    
    output_frames = []
    initial_frame = int(list(labels.keys())[0])
    for frame_id, (frame,label) in enumerate(labels.items()):
        if save:
            img = cv2.imread(frames_path[int(frame)-1])
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
                
                if save:  
                    cv2.rectangle(img, (int(detection['bbox'][0]), int(detection['bbox'][1])), (int(detection['bbox'][2]), int(detection['bbox'][3])), (0, 0, 255), 2)
                    cv2.putText(img, str(id), (int(detection['bbox'][0]), int(detection['bbox'][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                update_labels(labels, int(frame), id, detection['bbox'][0], detection['bbox'][1],
                              detection['bbox'][2] - detection['bbox'][0], detection['bbox'][3] - detection['bbox'][1], detection['confidence'])
            

            if save: 
                cv2.imwrite(f'overlap_frames/{frame_id:04}.jpg', img)
                if int(frame) - initial_frame < 60:
                    print(frame_id)
                    output_frames.append(Image.fromarray(img[:,:,::-1]))
                else:
                    if gif and done: 
                        done = False
                        print('generating GIF...')
                        frame_one = output_frames[0]
                        frame_one.save(fp='overlap.gif', format="GIF", append_images=output_frames, save_all=True, duration=20, loop=0)
                        print("Done")

        past_label = label # Save current tracks to use them in the next iteration
        

                
    
def metrics(ground_truth,detections): # Task 2.3
    acc = mm.MOTAccumulator(auto_id=True)
    
    for key in detections.keys():
        
        gt_centers = [((bbox['bbox'][0] + bbox['bbox'][2]) / 2, (bbox['bbox'][1] + bbox['bbox'][3]) / 2) for bbox in ground_truth[key]]
        gt_index = [int(gt['id']) for gt in ground_truth[key]]
            
        det_centers = [((bbox['bbox'][0] + bbox['bbox'][2]) / 2, (bbox['bbox'][1] + bbox['bbox'][3]) / 2) for bbox in detections[key]]
        det_index = [int(det['id']) for det in detections[key]]
            
        acc.update(
            gt_index,                     # Ground truth objects in this frame
            det_index,             # Detector hypotheses in this frame
            mm.distances.norm2squared_matrix(gt_centers, det_centers)
        )
        
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['idp', 'idr','idf1','recall','precision'], name='acc')
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
    model_name = 'retinanet_R_101_FPN_3x.yaml'
    #model_name = 'faster_rcnn_X_101_32x8d_FPN_3x.yaml'
    real_model_name = model_name.replace('.yaml', '')
    #path_detections = 'fine_tune'
    path_detections = 'off_the_shelve'

    # load labels
    detections = load_labels(path_detections, real_model_name + '.txt')
    
    frames_paths = get_frames_paths(path_video)
    tracking_overlap(detections,frames_paths,save=False,gif=False)
    
    path = 'tracking_by_overlap/' + path_detections
    write_predictions(path, detections, real_model_name)
    
    detections = load_labels(path, real_model_name + '.txt')
    print("\n")
    print(model_name)
    print(path_detections)
    metrics(ground_truth,detections) # Task 2.3

    print('finished')