import logging
from typing import List, Dict
import os 
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
from .dataset import Dataset
from ..utils import logging_util
from ..utils.file_utils import read_from_image,convert_yolo_to_coco, read_text_file,bbox_to_segmentation,is_dir_check
import json


class YoloToCocoConverter:
    def __init__(self, text_path: str,  label_file_path:str,\
                  path_to_image: str, save_json_path: str, logger_output_dir: str) -> None:
        self.logger = logging_util.initialize_logging(logger_output_dir)
        self.text_path = text_path
        self.path_to_image = path_to_image
        self.save_json_path =save_json_path
        self.logger_output_dir = logger_output_dir
        self.label_file_path = label_file_path
        self.labels = read_text_file(self.label_file_path).splitlines()
        # first thing to check is we need to have exactly same length of images and text files 
        assert len(os.listdir(self.text_path)) == len(os.listdir(self.path_to_image)), self.logger.error("Error! Please check the files & image length don't match")
        # second we check if the files have same name of images 
        assert set([img.split(".")[0] for img in os.listdir(self.path_to_image)]) == set([file.split(".")[0] for file in os.listdir(self.text_path)]), self.logger.error("Error! some files names don't match.")
        self.coco = Coco()
        self.converter()

    
    def converter(self):
        is_dir_check([self.save_json_path])
        allImages = os.listdir(self.path_to_image)
        cid_to_label =dict()
        for img in allImages:
            imgpath = os.path.join(self.path_to_image, img)
            _, height, width = read_from_image(imgpath)
            annotations_file = img.split(".")[0]+".txt"
            annotations  = read_text_file(os.path.join(self.text_path, annotations_file)).splitlines()
            cocimg = CocoImage(file_name=img, height=height, width=width)
            for annt in annotations:
                category_id, x_center, y_center, w, h = annt.split()
                label = self.labels[int(category_id)]
                bbox_coco =convert_yolo_to_coco([x_center, y_center, w, h], width, height)
                cocimg.add_annotation(CocoAnnotation(bbox=bbox_coco, \
                                                    category_id=category_id, \
                                                    category_name=label, \
                                                    segmentation= [bbox_to_segmentation(bbox_coco)]
                                        ))
                cid_to_label[category_id] = label
            self.coco.add_image(cocimg)
        for category_id, label in cid_to_label.items():
            self.coco.add_category(CocoCategory(id=category_id, name=label))
        # let's save the fies 
        save_json(self.coco.json, save_path=os.path.join(self.save_json_path,"coco.json"), indent=4)