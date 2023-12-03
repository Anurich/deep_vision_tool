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




class CocoToYoloConverter(Dataset):
    def __init__(self, json_data: Dict[str, any], path_to_image: str, save_json_path: str, logger_output_dir:str) -> None:
        super().__init__(json_data, path_to_image, save_json_path, logger_output_dir)
        self.logger = logging_util.initialize_logging(self.logger_output_dir)
        if self.is_coco_format(json_data) != True:
            self.logger.error("Error! file is not in COCO format.")
        else:
            self.convert()



    def convert(self):
        images = self.json_data["images"]
        categories = self.json_data["categories"]
        allcategories =list(map(lambda x: x["name"], categories))
        self.logger.info(f"Categories: {allcategories}")
        path_of_img_to_save = os.path.join(self.save_json_path, "images")
        labels_path = os.path.join(self.save_json_path, "labels")
        is_dir_check([self.save_json_path, path_of_img_to_save, labels_path])
        save_categories(allcategories, self.save_json_path)
        for img in images:
            imgId = img["id"]
            annotations = list(filter(lambda x: x["image_id"] == imgId, self.json_data["annotations"]))
            imgname = img["file_name"]
            imgpath = os.path.join(self.path_to_image,imgname)
            img, height, width = read_from_image(imgpath)
            for annt in annotations:
                category_id =annt["category_id"]
                bbox = annt["bbox"] # bbox is already in coco format we simply need to normalize it 
                bbox = yolo_normalization(convert_bbox_to_yolo_bbox(bbox,is_coco=True), height, width)
                x_center_norm, y_center_norm, x_norm, y_norm = bbox
                records = f"{category_id} {x_center_norm} {y_center_norm} {x_norm} {y_norm}"
                write_to_text(os.path.join(labels_path, imgname.split(".")[0]+".txt"), records)
                save_img(os.path.join(path_of_img_to_save, imgname), np.array(img))
        self.logger.info("Successfully convered COCO to YOLO file format.")


    
    def is_coco_format(self, data: Dict[str, any]) ->  bool:
        # Check for the presence of required keys
        required_keys = {'annotations', 'images', 'categories'}
        if not all(key in data for key in required_keys):
            return False

        # Check the structure of the 'annotations' key
        if not isinstance(data['annotations'], list):
            return False

        for annotation in data['annotations']:
            # Check for the presence of required keys in each annotation
            annotation_keys = {'image_id', 'category_id', 'bbox'}
            if not all(key in annotation for key in annotation_keys):
                return False

            # Check the structure of the bounding box
            if not isinstance(annotation['bbox'], list) or len(annotation['bbox']) != 4:
                return False

        # Check the structure of the 'images' key
        if not isinstance(data['images'], list):
            return False

        # Check the structure of the 'categories' key
        if not isinstance(data['categories'], list):
            return False

        return True


    def __repr__(self) -> str:
        return super().__repr__()
    
