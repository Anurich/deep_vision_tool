"""
    The file is responsible to convert the dataset into COCO.
    Library Used for this 
    # pip install sahi
"""
import logging
from typing import List, Dict
import os 
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
from .dataset import Dataset
from ..utils import logging_util
from ..utils.file_utils import get_all_categories, read_from_image,convert_bbox_to_coco_bbox, save_categories,is_dir_check,bbox_to_segmentation, convert_to_classification_format
import json
"""
[
    {
        "image_id": 1,
        "img_name": "image1.jpg",
        "annotations": [
            {
                "label": labelname,
                "bbox": [x1, y1, x2, y2],
                "segmentation": [...],
                "area": area
            },
            {
                "label": labelname,
                "bbox": [x1, y1, x2, y2],
                "segmentation": [...],
                "area": area
            },
            // Add more annotations as needed
        ]
    }
]
This is the format expected from this function in order to create a coco file.
"""
class CocoConverter(Dataset):
    def __init__(self, json_data: List[Dict[str, any]], path_to_image: str, save_json_path: str, logger_output_dir:str, type:str="object_detection") -> None:
        super().__init__(json_data, path_to_image, save_json_path, logger_output_dir, type)
        self.logger = logging_util.initialize_logging(self.logger_output_dir)
        # if it's classification we simply add 0,0,0,0 in bboxes and segmentation  
        if self.type=="classification":
            json_data = convert_to_classification_format(json_data=json_data)
        
        if not self.is_valid_json_structure(json_data):
            self.logger.error("Error! Please check the file structure")
        else:
            self.coco = Coco()
            self.all_categories = get_all_categories(self.json_data)
            if isinstance(self.all_categories, json.JSONDecodeError):
                self.logger.error(f"Json Error {self.all_categories}")
            else:
                self.logger.info("Starting Conversion To COCO....")
                is_dir_check([self.save_json_path])
                save_categories(self.all_categories, self.save_json_path)
                
                self.convert(self.all_categories)

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

    def convert(self, categories_name: List[str]):
        # creating the category of coco
        self.logger.info(f"Categories: {categories_name}")
        category_dict = {}
        for idx, category in enumerate(categories_name):
            category_dict[category] = idx
            self.coco.add_category(CocoCategory(id=idx, name=category))
        for dd in self.json_data:
            _, height, width = read_from_image(os.path.join(self.path_to_image, dd["img_name"]))
            cocimg = CocoImage(file_name=dd["img_name"], height=height, width=width)
            for annt in dd["annotations"]:
                converted_bbox_to_coco = convert_bbox_to_coco_bbox(annt["bbox"], type=self.type)
                cocimg.add_annotation(
                    CocoAnnotation(
                        bbox=converted_bbox_to_coco,
                        category_id=category_dict[annt["label"]],
                        category_name=annt["label"],
                        segmentation= [bbox_to_segmentation(converted_bbox_to_coco)],
                    )
                )

            self.coco.add_image(cocimg)
        self.logger.info("Successfully created COCO file")
        save_json(data=self.coco.json,save_path=os.path.join(self.save_json_path,"coco.json"), indent=4)

    def __repr__(self) -> str:
        return super().__repr__()