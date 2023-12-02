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
from ..utils.file_utils import get_all_categories, save_to_json, read_from_image,convert_bbox_to_coco_bbox
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
class ConvertToCoco(Dataset):
    def __init__(self, json_data: List[Dict[str, any]], path_to_image: str, save_json_path: str, logger_output_dir:str, type_of_data_converstion:str) -> None:
        super().__init__(json_data, path_to_image, save_json_path, logger_output_dir, type_of_data_converstion)
        self.logger = logging_util.initialize_logging(self.logger_output_dir)
        self.type_of_data_converstion = type_of_data_converstion
        if self.type_of_data_converstion.upper() == "COCO":
            self.coco = Coco()
            self.all_categories = get_all_categories(self.json_data)
            if isinstance(self.all_categories, json.JSONDecodeError):
                self.logger.error(f"Json Error {self.all_categories}")
            else:
                self.coco_construction(self.all_categories)
        else:
            pass

    def coco_construction(self, categories_name: List[str]):
        # creating the category of coco
        category_dict = {}
        for idx, category in enumerate(categories_name):
            category_dict[category] = idx
            self.coco.add_category(CocoCategory(id=idx, name=category))
        for dd in self.json_data:
            _, height, width = read_from_image(os.path.join(self.path_to_image, dd["img_name"]))
            cocimg = CocoImage(file_name=dd["img_name"], height=height, width=width)
            for annt in dd["annotations"]:
                converted_bbox_to_coco = convert_bbox_to_coco_bbox(annt["bbox"])
                cocimg.add_annotation(
                    CocoAnnotation(
                        bbox=converted_bbox_to_coco,
                        category_id=category_dict[annt["label"]],
                        category_name=annt["label"],

                    )
                )

            self.coco.add_image(cocimg)
        self.logger.info("Successfully created COCO file")
        save_json(data=self.coco.json,save_path=os.path.join(self.save_json_path,"coco.json"))

    def __repr__(self) -> str:
        return super().__repr__()