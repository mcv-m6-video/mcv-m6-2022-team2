import cv2
import os

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer

from utils.register_dataset import register_city_challenge

MODEL_ID = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml" # Faster R-CNN (X101-FPN)

if __name__=="__main__":

    # Raw model name
    model_name = MODEL_ID.replace('.yaml', '').split('/')[-1]  # Model name without .yml and COCO-Detection

    # Obtain the dataset
    test_dictionary = register_city_challenge('splits/test.txt')

    for text_files, split in zip(['splits/test.txt'], ['test']):
        DatasetCatalog.register("CITY_CHALLENGE_" + split, lambda split=split: register_city_challenge(text_files))
        MetadataCatalog.get("CITY_CHALLENGE_" + split).set(thing_classes=["car"])

    # --- CONFIGURATIONS ---
    # Model config with the saved weights
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_ID))  # model
    cfg.MODEL.WEIGHTS = os.path.join('models', model_name + '.pth')
    cfg.DATASETS.TEST = ("CITY_CHALLENGE_test",)

    # Create the predictor
    predictor = DefaultPredictor(cfg)

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
    for idx, dict in enumerate(test_dictionary):
        print(idx, dict['file_name'])
        # Load image
        img = cv2.imread(dict['file_name'])[:, :, ::-1]  # BGR to RGB
        outputs = predictor(img)

        v = Visualizer(img, MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)

        for cidx, bbox in enumerate(outputs["instances"].pred_boxes.tensor):
            v.draw_box(bbox.cpu(), alpha=0.9)

        cv2.imshow("frame",v.get_output().get_image())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # remove the colors of unsegmented pixels


