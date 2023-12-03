import logging
from typing import List, Dict
import os 
from PIL import Image
import numpy as np
from .dataset import Dataset
from ..utils import logging_util
from ..utils.file_utils import  get_all_categories, read_from_image,convert_bbox_to_yolo_bbox,\
                        yolo_normalization, write_to_text,is_dir_check, save_img, read_from_image,save_categories
import json

class YOLOConverter(Dataset):
    def __init__(self, json_data: List[Dict[str, any]], path_to_image: str, save_json_path: str, logger_output_dir:str) -> None:
        super().__init__(json_data, path_to_image, save_json_path, logger_output_dir)
        self.logger = logging_util.initialize_logging(self.logger_output_dir)
        if not self.is_valid_json_structure(self.json_data):
            self.logger.error("Error! Please check the file structure")
        else:
            self.all_categories = sorted(get_all_categories(self.json_data))
            if isinstance(self.all_categories, json.JSONDecodeError):
                self.logger.error(f"Json Error {self.all_categories}")
            else:
                self.logger.info("Starting Conversion To Yolo....")
                self.label2index = {key: idx for idx, key in enumerate(self.all_categories)}
                self.convert()



    def __repr__(self) -> str:
        return super().__repr__()

    def is_valid_json_structure(self,data):
        if not isinstance(data, list):
            return False
        entry = data[0]
        if not isinstance(entry, dict) or "image_id" not in entry or "img_name" not in entry or "annotations" not in entry:
            return False
        if not isinstance(entry["image_id"], int) or not isinstance(entry["img_name"], str) or not isinstance(entry["annotations"], list):
            return False
        for annotation in entry["annotations"]:
            if not isinstance(annotation, dict) or "label" not in annotation or "bbox" not in annotation or "segmentation" not in annotation or "area" not in annotation:
                return False
            if not isinstance(annotation["label"], str) or not isinstance(annotation["bbox"], list) or len(annotation["bbox"]) != 4 or not all(isinstance(coord, (int, float)) for coord in annotation["bbox"]) or not isinstance(annotation["segmentation"], list) or not isinstance(annotation["area"], (int, float)):
                return False
        return True

    def convert(self):
        """
            |converted_data
                |images
                    |im.png
                |labels
                    |im.txt
            
            # make sure to follow this file structure is followed
        """
        path_of_img_to_save = os.path.join(self.save_json_path, "images")
        labels_path = os.path.join(self.save_json_path, "labels")
        is_dir_check([self.save_json_path, path_of_img_to_save, labels_path])
        save_categories(self.all_categories, self.save_json_path)
        self.logger.info(f"Categories: {self.all_categories}")
        for data in self.json_data:
            imgname = data["img_name"]
            annotations = data["annotations"]
            imgpath = os.path.join(self.path_to_image,imgname)
            img, height, width = read_from_image(imgpath)
            for annt in annotations:
                label = annt["label"]
                category_id  = self.label2index[label]
                bbox = annt["bbox"]
                bbox = yolo_normalization(convert_bbox_to_yolo_bbox(bbox), height, width)
                x_center_norm, y_center_norm, x_norm, y_norm = bbox
                records = f"{category_id} {x_center_norm} {y_center_norm} {x_norm} {y_norm}"
                write_to_text(os.path.join(labels_path, imgname.split(".")[0]+".txt"), records)
                save_img(os.path.join(path_of_img_to_save, imgname), np.array(img))
        self.logger.info("Successfully created YOLO file")