import uuid
from dataset_gestions import load_labels
from dataset_gestions import update_labels, write_predictions
import numpy as np
from metric_functions import iou_from_list_of_ground_truths

def tracking_overlap(labels):

    initialize = True
    for frame, label in labels.items():
        if initialize:
            for detection in label:
                id = create_id()
                update_labels(labels, int(frame), id, detection['bbox'][0], detection['bbox'][1],
                              detection['bbox'][2], detection['bbox'][3], detection['confidence'])
            initialize = False
        else:
            gt_bboxes = np.zeros((len(past_label), 4))
            for idx, past_detections in enumerate(past_label):
                gt_bboxes[idx, :] = np.array((past_detections['bbox'][0], past_detections['bbox'][1],
                                              past_detections['bbox'][0] + past_detections['bbox'][2],
                                              past_detections['bbox'][1] + past_detections['bbox'][3]))
            for detection in label:
                bbox = [detection['bbox'][0], detection['bbox'][1],
                        detection['bbox'][0] + detection['bbox'][2],
                        detection['bbox'][1] + detection['bbox'][3]]

                ious = iou_from_list_of_ground_truths(gt_bboxes, bbox)

                if max(ious) > 0.5:
                    id = past_label[np.argmax(ious)]['id']

                else:
                    id = create_id()

                update_labels(labels, int(frame), id, detection['bbox'][0], detection['bbox'][1],
                              detection['bbox'][2], detection['bbox'][3], detection['confidence'])
        past_label = label


def tracking_kalman():
    print('todo')


def create_id():
    """
    creates unique identifier
    :return: identifier
    """
    return uuid.uuid4()


if __name__ == "__main__":

    # ........TESTING IF TASK 2.1 WORKS........

    # paths to models
    model_name = 'faster_rcnn_X_101_32x8d_FPN_3x.yaml'
    real_model_name = model_name.replace('.yaml', '')
    path_detections = 'off_the_shelve'

    # load labels
    detections = load_labels(path_detections, real_model_name + '.txt')

    tracking_overlap(detections)

    write_predictions(detections, real_model_name)

    print('finished')