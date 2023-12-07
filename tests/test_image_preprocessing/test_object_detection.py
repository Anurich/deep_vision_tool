from conftest  import *
from deep_vision_tool.image_preprocessing.object_detection import ObjectDetection
import pytest
def test_obj(tmp_path):
    objimage = ObjectDetection("data_folder/yolo/","yolo", tmp_path, "data_folder/training_images/","logs")
    imgObject = objimage.get_yolo_image_object()

    #objimage.visualize_image_object(imgObject, plot="both", save_directory=tmp_path)
