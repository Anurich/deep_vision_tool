import pytest
from conftest  import *
from hair_style_recommendation.dataset_conversion.data_to_yolo import convertoYOLO
import pandas as pd 
from PIL import Image
import cv2
from hair_style_recommendation.utils.file_utils import read_from_json, apply_bbox_to_img

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

"""

@pytest.fixture
def make_dummy_data(tmp_path):
    path_to_data = "data_folder/train_solution_bounding_boxes (1).csv"
    df = pd.read_csv(path_to_data)
    imgname = None
    gindex  = None
    for img_name, group_index in df.groupby("image").groups.items():
        if len(group_index) > 1:
            imgname = img_name
            gindex  = group_index
    
    bboxes = []
    for index in gindex:
        x1, y1, x2, y2 = df.loc[index][["xmin", "ymin", "xmax", "ymax"]]
        bboxes.append([int(x1), int(y1), int(x2), int(y2)])

    json_data = [
        {
            "image_id": 1,
            "img_name": imgname,
            "annotations":[
                {
                    "label": "car",
                    "bbox": bboxes[0],
                },
                {
                    "label": "car",
                    "bbox": bboxes[1],
                }
            ]
        }
    ]

    tmppath = tmp_path / "tmp"
    tmppath.mkdir()
    return [json_data, "data_folder/training_images/", tmppath, "logs/"]


def test_convert_to_yolo(make_dummy_data):
    json_data, imgpath, savejsonpath, logpath =make_dummy_data
    convertoYOLO(json_data, imgpath, savejsonpath, logpath)
    img_path = os.path.join(savejsonpath, "images")
    label_path = os.path.join(savejsonpath, "labels")

    assert len(os.listdir(img_path)) == len(os.listdir(label_path))
    assert len(os.listdir(savejsonpath)) == 2

