import glob

import numpy as np
import cv2

from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

setup_logger()  # initialize the detectron2 logger and set its verbosity level to “DEBUG”.

def build_model(model_path) :
    confidence = 0.7 
    
    ## detectron2 Config setup ##
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))  # WAS 3x.y
    cfg.MODEL.WEIGHTS = model_path 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence
    # cfg.MODEL.DEVICE = 'cpu'
    predictor = DefaultPredictor(cfg)

    # predictions = predictor(image)

    return predictor

def draw_segmented_image(image, predictions, classes) :
    MetadataCatalog.get("user_data").set(thing_classes=classes)
    metadata = MetadataCatalog.get("user_data")
    v = Visualizer(image, metadata=metadata, instance_mode=ColorMode.IMAGE)
    out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
    image = out.get_image() 
    
    return image
    