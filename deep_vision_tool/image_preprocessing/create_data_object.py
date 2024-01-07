
from ..image_object.annotation import Annotation
from ..image_object.Image import ImageInfo
from typing import Dict, List, Any
import numpy as np
import pandas as pd 
from ..utils.file_utils import read_from_json, read_from_image, is_coco_format, apply_bbox_to_img, \
      visualize_and_save, visualize_segmentation_mask, is_dir_check,store_pickle, read_text_file, convert_yolo_to_coco
import os
from ..utils.logging_util import initialize_logging
import logging

class ObjectDetection:
    def __init__(self, filepath: str,  type_of_data: str, \
                save_image_object_path:str, \
                image_path: str=None, log_dir:str=None, type:str="object_detection") -> None:
        """
            Initialize the data handler.

            Parameters:
            - filepath (str): The path to the data file.
            - type_of_data (str): The type of data, either "yolo" or other.
            - save_image_object_path (str): The path to save image objects.
            - image_path (str, optional): The path to the image directory. Defaults to None.
            - log_dir (str, optional): The directory for log files. Defaults to None.

            If `type_of_data` is "yolo", provide the path to the YOLO data file using `filepath`.
            For other types of data, ensure to provide the necessary paths and configurations.

            Attributes:
            - logger: Logger object for logging information.
            - filepath (str): The path to the data file.
            - type_of_data (str): The type of data.
            - image_path (str): The path to the image directory, if applicable.
            - save_image_object_path (str): The path to save image objects.
            - log_dir (str): The directory for log files.
        """
        self.logger = initialize_logging(log_dir)
        self.filepath = filepath
        self.type = type
        self.type_of_data = type_of_data
        self.image_path = image_path
        self.save_image_object_path = save_image_object_path
        # we have currently two type of format that we cater 
        # 1 YOLO
        # 2 COCO
    def get_yolo_image_object(self) -> List[ImageInfo]:
        assert self.type_of_data.lower() =="yolo", self.logger.error("they type of data is not yolo, please use type_of_data as yolo")
        yolo_image_obj=[]
        allFolders = os.listdir(self.filepath) # it should contain two folders images, and labels
        
        assert "images" in allFolders, self.logger.error("No images folder found!")
        assert "labels" in allFolders, self.logger.error("No labels folder found!")
        assert os.path.isfile(os.path.join(self.filepath, "labels.txt")) == True, self.logger.error("Labels.txt is missing!")
        # constructing the paths
        image_path = os.path.join(self.filepath, "images")
        labels_path= os.path.join(self.filepath,"labels")
        assert len(os.listdir(image_path)) == len(os.listdir(labels_path)), self.logger.error("images and labels length don't match...")
        # assert [for im in ]
        category_label_file = os.path.join(self.filepath, "labels.txt")
        annt_ids = 0
        for idx, imname in enumerate(os.listdir(image_path)):
            assert imname.split(".")[0]+".txt" in os.listdir(labels_path)
            image_file_path = os.path.join(image_path, imname)
            _, height, width = read_from_image(image_file_path)

            labels_file_path = os.path.join(labels_path, imname.split(".")[0]+".txt")
            data = read_text_file(labels_file_path).splitlines()
            labels_list = read_text_file(category_label_file).splitlines()
            annotations = []
            for annts in (data):
                category_id, center_x, center_y, w, h  = annts.split() # yolo_labels
                annotations.append(Annotation(
                    image_id=idx, 
                    id=annt_ids,
                    bbox=[center_x, center_y, w, h],
                    segmentation=[], 
                    category_id=category_id, 
                    label=labels_list[int(category_id)]))
                annt_ids+=1

            
            img_object = ImageInfo(id=idx,
                                   filename=imname,
                                   image_path=image_path,
                                   im_width=width, 
                                   im_height=height, 
                                   annotations=annotations)

            yolo_image_obj.append(img_object)
        
        # now that we have the yolo object we can save this object
        self.logger.info("Successfully created yolo image object")
        is_dir_check([self.save_image_object_path])
        store_pickle(yolo_image_obj, self.save_image_object_path, "yolo_image_info.pickle")
        return yolo_image_obj


    def get_coco_image_object(self)-> List[ImageInfo]:
        """
        Perform pre-processing for COCO data.

        Returns:
        List[ImageInfo]: A list of ImageInfo objects containing image and annotation information.
        """
        assert self.type_of_data.lower() == "coco", self.logger.error("The type of data is not coco, please use type_of_data as coco")
        coco_image_obj = []
        coco_file = list(filter(lambda x: x=="coco.json",os.listdir(self.filepath)))[0]
        assert coco_file == "coco.json", self.logger.error("No coco.json file exist in this folder.")
        labels_path = os.path.join(self.filepath,"labels.txt")
        assert os.path.isfile(labels_path) == True, self.logger.error("labels.txt is missing !")
        labels = read_text_file(labels_path).splitlines() # reading the labels
        coco_data = read_from_json(os.path.join(self.filepath,coco_file))
        # we need to check also if the data is coco type 
        assert is_coco_format(coco_data,type=self.type) == True, self.logger.error("Not a valid coco data.")
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
            filtered_annotations = list(filter(lambda x: int(x["image_id"])==int(img_id), annotations))
            # now that we have filtered annotations we can now create an annotation object 
            annotations_object = []
            for annt in filtered_annotations:
                if self.type == "classification":
                    annt["bbox"] = [0]*4
                    annt["segmentation"] =[0]*8
                annotations_object.append(Annotation(
                    image_id=annt["image_id"],
                    id=annt["id"],
                    bbox=annt["bbox"],
                    segmentation=annt["segmentation"],
                    category_id=annt["category_id"],
                    label=labels[annt["category_id"]],
                ))
            # now that we have an image object and annotations object we can simply call super object with both image and annotations information
            img_object = ImageInfo(id=img_id, filename=img_filename,image_path=self.image_path, im_width=img_width, im_height=img_height, annotations=annotations_object)
            coco_image_obj.append(img_object)

        self.logger.info("Succesfully created image object.")
        is_dir_check([self.save_image_object_path])
        store_pickle(coco_image_obj, self.save_image_object_path,"coco_img_obj.pickle")
        return coco_image_obj


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
            is_dir_check([save_directory])
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
            labels =[]
            for annt in imginfo.annotations:
                if self.type_of_data == "yolo":
                    bboxes.append(convert_yolo_to_coco(annt.bbox,imginfo.im_width, imginfo.im_height))
                else:
                    bboxes.append(annt.bbox)

                segmentations.extend(annt.segmenation)
                labels.append(annt.label)

            # let's plot these annotations 
            if plot=="bbox":
                img_with_bbox = apply_bbox_to_img(bboxes, img, labels)
                results = visualize_and_save(img_with_bbox, save_directory, imgfile)
                if isinstance(results, str):
                    self.logger.warning(results)
            elif plot =="segmentation":
                results = visualize_segmentation_mask(segmentations, img,width, height, save_directory, imgfile, labels)
                if isinstance(results, str):
                    self.logger.warning(results)
            elif plot=="both":
                img_with_bbox = apply_bbox_to_img(bboxes, img.copy(), labels)
                results_with_bbox = visualize_and_save(img_with_bbox, save_directory, imgfile)
                results_with_segment = visualize_segmentation_mask(segmentations, img.copy(),width, height, save_directory, imgfile,labels)
                if isinstance(results_with_bbox, str):
                    self.logger.warning(results_with_bbox)
                if isinstance(results_with_segment, str):
                    self.logger.warning(results_with_segment)
        
        self.logger.info("Visualization finished")
        