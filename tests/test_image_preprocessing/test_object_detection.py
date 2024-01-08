from conftest  import *
from deep_vision_tool.image_preprocessing.create_data_object import ObjectDetection
from deep_vision_tool.image_object.annotation import Annotation
import pytest
def test_obj(tmp_path):
    objimage_yolo = ObjectDetection("data_folder/yolo/","yolo", tmp_path, "data_folder/training_images/","logs")
    objimage_coco = ObjectDetection("data_folder/coco/","coco", tmp_path, "data_folder/training_images/","logs")
    objimage_yolo.get_yolo_image_object()
    objimage_coco.get_coco_image_object()
    for iminfo in objimage_yolo.yolo_image_obj:
        assert isinstance(iminfo.filename, str)
        assert isinstance(iminfo.image_path, str)
        assert isinstance(iminfo.annotations[0], Annotation)
        assert isinstance(iminfo.im_width, int)
        assert isinstance(iminfo.im_height, int)
    for imginfo in objimage_coco.coco_image_obj:
        assert isinstance(iminfo.filename, str)
        assert isinstance(iminfo.image_path, str)
        assert isinstance(iminfo.annotations[0], Annotation)
        assert isinstance(iminfo.im_width, int)
        assert isinstance(iminfo.im_height, int)
