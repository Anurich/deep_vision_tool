from conftest  import *
from deep_vision_tool.image_preprocessing.object_detection import ObjectDetection
import pytest
def test_obj(tmp_path):
    print(tmp_path)
    objimage = ObjectDetection("data_folder/coco/","coco", "data_folder/training_images/","logs")
    imgObject = objimage.coco_postprocessing()
    objimage.visualize_image_object(imgObject, plot="both", save_directory=tmp_path)