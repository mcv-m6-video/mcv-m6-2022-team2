from register_dataset import register_city_challenge
from utils.image_utils import plotBBoxes
import cv2
import matplotlib.pyplot as plt

test_dictionary = register_city_challenge('splits/train.txt')

cv2.namedWindow("image", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
for idx, dict in enumerate(test_dictionary):

    im = cv2.imread(dict['file_name'])

    # Draw the frame number on the frame
    cv2.putText(im, str(idx), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    annotations = dict['annotations']
    # Obtain the frame BBoxes
    if len(annotations) > 0:
        annotations = [annotation['bbox'] for annotation in annotations]
        im = plotBBoxes(im, saveFrames=None, annotations=annotations)

    cv2.imshow('image', im)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break