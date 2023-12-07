from conftest  import *
from deep_vision_tool.image_preprocessing.object_detection import ObjectDetection
from deep_vision_tool.image_object.annotation import Annotation
import pytest
def test_obj(tmp_path):
    objimage_yolo = ObjectDetection("data_folder/yolo/","yolo", tmp_path, "data_folder/training_images/","logs")
    objimage_coco = ObjectDetection("data_folder/coco/","coco", tmp_path, "data_folder/training_images/","logs")
    imgObject = objimage_yolo.get_yolo_image_object()
    imgObject_coco = objimage_coco.get_coco_image_object()
    
    for iminfo in imgObject:
        assert isinstance(iminfo.filename, str)
        assert isinstance(iminfo.image_path, str)
        assert isinstance(iminfo.annotations[0], Annotation)
        assert isinstance(iminfo.im_width, int)
        assert isinstance(iminfo.im_height, int)
    for imginfo in imgObject_coco:
        assert isinstance(iminfo.filename, str)
        assert isinstance(iminfo.image_path, str)
        assert isinstance(iminfo.annotations[0], Annotation)
        assert isinstance(iminfo.im_width, int)
        assert isinstance(iminfo.im_height, int)
