"""
The idea in this .py is to create all the metrics needed in order to evaluate our detection systems:
IoU, mAP (several functions will be needed).
"""

import numpy as np
import os

def iou(boxA, boxB):
    """
    Compute IoU of two bounding boxes:
        0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)

    BBOX => [topLeft, bottomRight] = [xTL, yTL, xBR, yBR]
    :param boxA: BBOX 1
    :param boxB: BBOX 2
    :return: returns IoU: 0<=IoU<=1
    """
    # (x, y) coordinates of the intersection
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Area of intersection
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Area of prediction rectangle
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)

    # Area of ground-truth rectangle
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # IoU is computed by using the area and dividing it by the sum of prediction + ground-truth areas
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def iou_from_list_of_ground_truths(bboxes_gt, bbox):
    """
    Computes IoU from bboxes_gt that is a list and bbox that is a single array.
    The procedure is the same as the iou function before but now taking bboxes_gt as a list
    :param bboxes_gt: Ground truth bboxes
   (num, bbox) => num:number of boxes, bbox: [xTL, yTL, xBR, yBR]
    :param bbox: Detected bbox
   (bbox,) => bbox: [xTL, yTL, xBR, yBR]
    :return: All the iou's (size num)
    """

    # intersection
    xA = np.maximum(bboxes_gt[:, 0], bbox[0])
    yA = np.maximum(bboxes_gt[:, 1], bbox[1])
    xB = np.minimum(bboxes_gt[:, 2], bbox[2])
    yB = np.minimum(bboxes_gt[:, 3], bbox[3])
    iw = np.maximum(xB - xA + 1., 0.)
    ih = np.maximum(yB - yA + 1., 0.)
    inters = iw * ih

    # union
    uni = ((bbox[2] - bbox[0] + 1.) * (bbox[3] - bbox[1] + 1.) +
           (bboxes_gt[:, 2] - bboxes_gt[:, 0] + 1.) *
           (bboxes_gt[:, 3] - bboxes_gt[:, 1] + 1.) - inters)

    return inters / uni


def mean_iou(bboxes_gt_frame, bboxes_frame):
    """
    Compute mean iou by averaging the individual results on each iou (in a specific frame).
    :param bboxes_gt_frame: ground truth bounding box
    :param bboxes_frame: list of the detected bounding box for each frame
    :return: mean iou
    """
    iou = []
    for detection in bboxes_frame:
        iou.append(np.max(iou_from_list_of_ground_truths(bboxes_gt_frame, detection)))

    return np.mean(iou)


def compute_average_precision_voc(recall, precision):
    """
    Computes average precision using the 11 point method (VOC)
    code extracted from team 3 of last year: https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week1/metrics.py
    :param recall: Recall
    :param precision: Precision
    :return: average precision (area of precision recall curve)
    """
    average_precision = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        average_precision = average_precision + p / 11.
    return average_precision


def process_ground_truths(ground_truths, frames_index,class_name):
    """
    Extract ground truth objects for a class
    :param ground_truths: Dictionary where for each frame we have a list of dictionaries of the labels PD: debug to understand it better!
    :param frames_index: Index of starting frame in the ground truth dict.
    :param class_name: Name of the class, in this case, it will be always car
    :return:
    class_recs: Dictionary where: each frame is a dictionary, and inside it there are the bounding box for each detection and
    the variable det (has the information if the detection is true or false).
    npos: number of detections in gt
    """          

    class_recs = {}
    npos = 0
    for idx,ground_truth_frame in enumerate(ground_truths,frames_index):
        det = []
        bboxes = []
        for obj in ground_truth_frame:
            if obj['name'] == class_name:
                bboxes.append(obj['bbox'])
                det.append(False)
                npos += 1
                
        class_recs[f"{idx:04}"] = {'bbox': np.array(bboxes),
                                   'det': det}

    return class_recs, npos


def process_detections(detections, class_recs):
    """
    Process the detections and sort by confidence
    :param detections: Same as ground_truths but with the detections (output of a model)
    :param class_recs: Dictionary where: each frame is a dictionary, and inside it there are the bounding box for each detection and
    the variable det (has the information if the detection is true or false).
    :return:
    image_ids: id's of the frames (for each Bbox we have a id on the frame that is located)
    BB: array with all the BBoxes in all the video (sorted by confidence)
    """
    image_ids = [frame for frame, objs in detections.items() for _ in objs if frame in class_recs.keys()]
    confidence = np.array(
        [obj['confidence'] for frame, objs in detections.items() for obj in objs if frame in class_recs.keys()])
    BB = np.array([obj['bbox'] for frame, objs in detections.items() for obj in objs if frame in class_recs.keys()])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind]
    image_ids = [image_ids[x] for x in sorted_ind]

    return image_ids, BB


def process_results(image_ids, class_recs, BB, iou_threshold):
    """
    Cross the detections and the ground truths in order to get the results
    :param image_ids: id's of the frames (for each Bbox we have a id on the frame that is located)
    :param class_recs: Dictionary where: each frame is a dictionary, and inside it there are the bounding box for each detection and
    :param BB: array with all the BBoxes in all the video (sorted by confidence)
    :param iou_threshold: Criteria to define if the predictions are correct; if IoU < threshold the result will be considered as incorrect
    :return:
    fp: false positives
    fn: false negatives
    """
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            overlaps = iou_from_list_of_ground_truths(BBGT, bb)

            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > iou_threshold:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.
    return fp, tp


def compute_prec_and_recall(fp, tp, npos):
    """
    Compute precision and recall.
    :param fp: array of false positives (size of bounding boxes in all the frames)
    :param tp: array of true positives (size of bounding boxes in all the frames)
    :param npos: size of bounding boxes in all the frames
    :return:
    Prec, recall.
    """
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    return prec, rec


def evaluation_single_class(ground_truths, detections, frames_index, class_name='car', iou_threshold=0.5):
    """
    ################## MAIN FUNCTION! => Perform Pascal VOC evaluation ##########################
    code extracted (and refined!) from team 3 of last year: https://github.com/mcv-m6-video/mcv-m6-2021-team3/blob/main/Week1/metrics.py
    :param ground_truths: Dictionary where for each frame we have a list of dictionaries of the labels PD: debug to understand it better!
    :param frame_names: Path of the frames (where are located the images of the video).
    :param detections: Same as ground_truths but with the detections (output of a model)
    :param class_name: Name of the class, in this case, it will be always car
    :param iou_threshold: Criteria to define if the predictions are correct; if IoU < threshold the result will be considered as incorrect
    :return:
    precision, recall.
    average precision: area under precision recall curve
    """
    # extract ground_truth objects for this class
    class_recs, npos = process_ground_truths(ground_truths, frames_index, class_name)

    # Process detections and sort by confidence
    image_ids, BB = process_detections(detections, class_recs)

    # Process results (detections vs ground truths)
    fp, tp = process_results(image_ids, class_recs, BB, iou_threshold)

    # Compute precision and recall from true positives and false positives
    prec, rec = compute_prec_and_recall(fp, tp, npos)

    # Compute average precision (area on the Precision-Recall curve) with the precision and recall vectors
    ap = compute_average_precision_voc(rec, prec)

    return rec, prec, ap