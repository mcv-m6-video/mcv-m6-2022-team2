import os
from os.path import join
import xml.etree.ElementTree as ET
import glob
import subprocess
import cv2
from tqdm import tqdm
def update_labels(labels, frame_id, id, xmin, ymin, xmax, ymax, confidence):
    """
    This function uploades the labels.
    CAREFUL! if this function crashes, might be because id is missing in the anterior implementations...
    :param labels: dictionary of all the labels (annotations)
    :param frame_id: id of the frame
    :param id: unique identifier of the object
    :param xmin: coordinate x top left bbox
    :param ymin: coordinate y top left bbox
    :param xmax: coordinate x bottom right bbox
    :param ymax: coordinate y bottom right bbox
    :param confidence: confidence of the detection
    :return: labels: dictionary of all the labels uploaded with the parameters in question
    """
    frame_name = '%04d' % int(frame_id)
    obj_info = dict(
        name='car',
        bbox=[xmin, ymin, xmax, ymax],
        confidence=confidence,
        id=id
    )

    if frame_name not in labels.keys():
        labels.update({frame_name: [obj_info]})
    else:
        # actualizamos id's modo gitano activado
        append=True
        for idx, detections in enumerate(labels[frame_name]):
            if detections['bbox'][0] == xmin and detections['bbox'][1] == ymin:
                labels[frame_name][idx].update(obj_info)
                append=False

        if append:
            labels[frame_name].append(obj_info)

    return labels


def load_labels(path, name):
    """
    Load annotations. Supported files .xml and .txt.
    Annotations are a list of several dictionaries (debug with more info). In each dictionary, there are the
    names of the labels, the bboxes, confidence of the detections (more info will be added in next implementations)
    .txt case:
    < frame >, < id >, < bb_left >, < bb_top >, < bb_width >, < bb_height >, < conf >, < x >, < y >, < z >
    :param path: (string) path to the file where are all the annotations
    :param name: (string) name of the file where are all the annotations
    :return: labels: (dict) parsed annotations
    """
    # For the .txt case
    if name.endswith('txt'):
        with open(join(path, name), 'r') as f:
            txt = f.readlines()

        labels = {}
        for frame in txt:
            frame_id, id, xmin, ymin, width, height, confidence, _, _, _ = list(
                map(float, (frame.split('\n')[0]).split(',')))
            update_labels(labels, int(frame_id), id, xmin, ymin, xmin + width, ymin + height, confidence)

        return labels

    # For the .xml case
    elif name.endswith('xml'):
        tree = ET.parse(join(path, name))
        root = tree.getroot()
        labels = {}

        for child in root:
            if child.tag in 'track':
                # Only take into account 'cars'
                if child.attrib['label'] not in 'car':
                    continue
                
                id = child.attrib['id']
                for bbox in list(child):
                    frame_id, xmin, ymin, xmax, ymax, _, _, _ = list(map(float, ([v for k, v in bbox.attrib.items()])))
                    update_labels(labels, int(frame_id) + 1, id, xmin, ymin, xmax, ymax, 1.)

        return labels

    # Other formats are not supported
    else:
        assert 'Not supported format in the annotation file or other error (make sure that the path to the labels is ' \
               'correct). '


# Before, generation of the video with ffmpeg is needed:
# Install ffmpeg: https://es.wikihow.com/instalar-FFmpeg-en-Windows

# https://stackoverflow.com/questions/10957412/fastest-way-to-extract-frames-using-ffmpeg
# to extract one fps
# command: ffmpeg -i vdo.avi -r 1/1 vdo/$filename%03d.png
# to extract all frames (as the video is 10 fps)
# command: ffmpeg -i vdo.avi  -r 10/1 vdo/$filename%03d.png

def get_frames_paths(path_frames):
    """
    Get the paths where all the frames of the video are located.
    :param path_frames: (string) direction where the video is located
    :return: list of the paths of the frames (in order)
    """
    if not os.path.exists(path_frames):
        print('Creating the vdo folder with all the frames of the video...')
        os.makedirs(path_frames)                                                # Create vdo folder
        parent_dir = os.path.abspath(os.path.join(path_frames, os.pardir))      # Obtain parent directory
        capture = cv2.VideoCapture(f'{parent_dir}/vdo.avi')                     # Create Capture object
        n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))                   # Obtain number of frames
        frame_counter = 1                                                       # Variable to count frames
        loop = tqdm(range(n_frames), total=n_frames)                            # Progress bar

        while capture.isOpened():                                               # Loop through the video frames
            ret, frame = capture.read()                                         # Obtain frames

            if ret:
                cv2.imwrite(f'{path_frames}/{frame_counter:04}.png', frame)     # Save the image frame
                frame_counter += 1                                              # Update counter
                loop.update(1)                                                  # Update progress bar
            else:
                print("End of Video")                                           # Video is finished
                break

        capture.release()

    images_paths = glob.glob(join(path_frames, '*.png'))                        # Obtain the paths of all the images
    images_paths.sort()

    return images_paths


def write_predictions(path,labels, model):
    """
    writes predictions from labels dictionary into a .txt
    :param labels: labels dictionary of annotations
    :param model: name of the model used to do inference
    :return: -
    """

    os.makedirs(path, exist_ok=True)

    print('writting predictions into a txt file...')
    with open(path + '/' + model + ".txt", "w") as file:
        for label in labels.items():
            for detection in label[1]:
                bbox = detection['bbox']
                conf = detection['confidence']
                id = detection['id']
                # frame_id, id_detection (-1 bc only cars), bboxes, conf, x, y, z
                #file.write(f'{int(label[0])+1},{id},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{conf},-1,-1,-1\n')
                file.write(f'{int(label[0])},{id},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{conf},-1,-1,-1\n')


    print('predictions written into a txt file...')