import pytest
from conftest  import *
from deep_vision_tool.dataset_conversion.yolo_to_coco import YoloToCocoConverter
import pandas as pd 
from PIL import Image
import cv2
from deep_vision_tool.utils.file_utils import read_from_json, apply_bbox_to_img

@pytest.fixture
def dummy_data(tmp_path):
    tmppath = tmp_path / "yolo_to_coco"
    tmppath.mkdir()
    return [
        "data_folder/yolo/labels/",
        "data_folder/yolo/labels.txt",
        "data_folder/yolo/images",
        tmppath,
        "logs/"
    ]


def test_yolo_to_coco_converter(dummy_data):
    text_path, label_path, image_path, save_directory, log_dir = dummy_data
    YoloToCocoConverter(text_path, label_path, image_path, save_directory, log_dir)
