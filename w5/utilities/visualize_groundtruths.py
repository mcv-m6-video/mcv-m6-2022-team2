import cv2
import time
from dataset_utils import load_annot
from image_utils import plotBBoxes
from os.path import join

DATA_ROOT = '../../../data/AICity_data/train'
SEQ = 'S04'
CAM = 'c034'

labels = load_annot(join(DATA_ROOT, SEQ, CAM, 'gt'), 'gt.txt')
video = cv2.VideoCapture(join(DATA_ROOT, SEQ, CAM, 'vdo.avi'))

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions

count = 1
while True:
    ret, frame = video.read()

    if ret:
        # Draw the frame number on the frame
        cv2.putText(frame, str(count), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # Obtain the frame BBoxes
        annotations = labels.get(f'{count:04}',[])   # To avoid crashing when the frame does not contain annotations
        if len(annotations) > 0:
            annotations = [annotation['bbox'] for annotation in annotations]
            plotBBoxes(frame, saveFrames=None, annotations=annotations)

        cv2.imshow('frame', frame)
        time.sleep(0.05)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break