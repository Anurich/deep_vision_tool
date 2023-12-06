
from ..image_object.coco_annotation import CocoAnnotation
from ..image_object.yolo_annotation import YoloAnnotation
from ..image_object.ImageObjectCoco import ImageInfo
from typing import Dict, List, Any
import numpy as np
import pandas as pd 
from ..utils.file_utils import read_from_json, read_from_image, is_coco_format, apply_bbox_to_img, \
      visualize_and_save, visualize_segmentation_mask
import os
from ..utils.logging_util import initialize_logging
import logging

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
       
    def coco_postprocessing(self)-> List[ImageInfo]:
        """
        Perform post-processing for COCO data.

        Returns:
        List[ImageInfo]: A list of ImageInfo objects containing image and annotation information.
        """
        assert self.type_of_data.lower() == "coco", self.logger.error("The type of data is not coco")
        postprocessed_coco_image_annotations = []
        coco_file = list(filter(lambda x: x=="coco.json",os.listdir(self.filepath)))[0]
        coco_data = read_from_json(os.path.join(self.filepath,coco_file))
        # we need to check also if the data is coco type 
        assert is_coco_format(coco_data) == True, self.logger.error("Not a valid coco data.")
        # now that we have coco data we can create and object containing the image object and annotations 
        images = coco_data["images"]
        annotations = coco_data["annotations"]
        self.logger.info(f"Total number of images {len(images)}")
        self.logger.info(f"Total number of annotations {len(annotations)}")
        self.logger.info("Creating image object information form coco dataset.....")
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
        
        self.logger.info("Succesfully created image object.")
        return postprocessed_coco_image_annotations
            

    def visualize_image_object(self, image_info_object: List[ImageInfo], total_images: int =1, plot:str="both", save_directory: str=None):
        """
        Visualize a list of ImageInfo objects.

        Parameters:
        - image_info_objects (List[ImageInfo]): A list of ImageInfo objects.
        - total_images (int): The total number of images to visualize.
        - plot (str): Specify what to plot, options: "both", "bbox", "segmentation".

        """
        # total number of image object 
        if save_directory!=None:
            if not os.path.isdir(save_directory):
                os.mkdir(save_directory)

        self.logger.info(f"Visualising the total {total_images} images")
        for imginfo in image_info_object[:total_images]:
            imgfile = imginfo.filename
            imgpath = imginfo.image_path
            img_file_path = os.path.join(imgpath, imgfile)
            img, height, width = read_from_image(img_file_path)
            self.logger.info(f"Image shape {img.shape}, Width {width} Height {height}")
            # now we can take tha annotations 
            bboxes = []
            segmentations = []
            for annt in imginfo.annotations:
               bboxes.append(annt.bbox)
               segmentations.extend(annt.segmenation)

            # let's plot these annotations 
            if plot=="bbox":
                img_with_bbox = apply_bbox_to_img(bboxes, img)
                results = visualize_and_save(img_with_bbox, save_directory, imgfile)
                if isinstance(results, str):
                    self.logger.warning(results)
            elif plot =="segmentation":
                results = visualize_segmentation_mask(segmentations, img,width, height, save_directory, imgfile)
                if isinstance(results, str):
                    self.logger.warning(results)
            elif plot=="both":
                img_with_bbox = apply_bbox_to_img(bboxes, img.copy())
                results_with_bbox = visualize_and_save(img_with_bbox, save_directory, imgfile)
                results_with_segment = visualize_segmentation_mask(segmentations, img.copy(),width, height, save_directory, imgfile)
                if isinstance(results_with_bbox, str):
                    self.logger.warning(results_with_bbox)
                if isinstance(results_with_segment, str):
                    self.logger.warning(results_with_segment)
        self.logger.info("Visualization finished")