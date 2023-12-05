
from ..image_object.coco_annotation import CocoAnnotation
from ..image_object.yolo_annotation import YoloAnnotation
from ..image_object.ImageObjectCoco import ImageInfo
from typing import Dict, List, Any
from ..utils.file_utils import read_from_json
import os
from ..utils.logging_util import initialize_logging
class ObjectDetection:
    def __init__(self, filepath: str,  type_of_data: str, image_path: str=None, log_dir:str=None) -> None:
        """
            1. if type_of_data == "yolo" simply pass the 
        """
        self.logger = initialize_logging(log_dir)
        self.filepath = filepath
        self.type_of_data = type_of_data
        self.image_path = image_path
        # we have currently two type of format that we cater 
        # 1 YOLO
        # 2 COCO
       
    def coco_postprocessing(self):
        assert self.type_of_data.lower() == "coco"
        postprocessed_coco_image_annotations = []
        coco_file = list(filter(lambda x: x=="coco.json",os.listdir(self.filepath)))[0]
        coco_data = read_from_json(os.path.join(self.filepath,coco_file))
        # now that we have coco data we can create and object containing the image object and annotations 
        images = coco_data["images"]
        annotations = coco_data["annotations"]
        # now we can create an annotations object 
        
        for img in images:
            img_id = img["id"]
            img_filename = img["file_name"]
            img_width = img["width"]
            img_height = img["height"]
            filtered_annotations = (filter(lambda x: int(x["image_id"])==int(img_id), annotations))
            # now that we have filtered annotations we can now create an annotation object 
            annotations = []
            for annt in filtered_annotations:
                annotations.append(CocoAnnotation(
                    image_id=annt["image_id"],
                    id=annt["id"],
                    bbox=annt["bbox"],
                    segmentation=annt["segmentation"],
                    category_id=annt["category_id"]
                ))
            
            # now that we have an image object and annotations object we can simply call super object with both image and annotations information
            img_object = ImageInfo(id=img_id, filename=img_filename,image_path=self.image_path, im_width=img_width, im_height=img_height, annotations=annotations)
            postprocessed_coco_image_annotations.append(img_object)
        return postprocessed_coco_image_annotations
            

    def check_stats(self):
        pass
    