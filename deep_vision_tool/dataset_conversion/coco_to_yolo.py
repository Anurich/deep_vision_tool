import logging
from typing import List, Dict
import os 
from PIL import Image
import numpy as np
from .dataset import Dataset
from ..utils import logging_util
from ..utils.file_utils import  get_all_categories, read_from_image,convert_bbox_to_yolo_bbox,\
                        yolo_normalization, write_to_text,is_dir_check, save_img, read_from_image,save_categories, is_coco_format
import json




class CocoToYoloConverter(Dataset):
    def __init__(self, json_data: List[Dict[str, any]], path_to_image: str, save_json_path: str, logger_output_dir:str) -> None:
        super().__init__(json_data, path_to_image, save_json_path, logger_output_dir)
        self.logger = logging_util.initialize_logging(self.logger_output_dir)
        if is_coco_format(json_data) != True:
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

    def __repr__(self) -> str:
        return super().__repr__()
    
