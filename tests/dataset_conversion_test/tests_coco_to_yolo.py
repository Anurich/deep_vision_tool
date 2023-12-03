import pytest
from conftest  import *
from deep_vision_tool.dataset_conversion.coco_to_yolo import CocoToYoloConverter

@pytest.fixture
def dummy_coco(tmp_path):
    # we need to get the 
    tmppath = tmp_path / "tmp"
    tmppath.mkdir()
    return [{
        "images":[
            {
                "height":380,
                "width":676,
                "id":1,
                "file_name":"vid_4_720.jpg"
            }
        ],
        "annotations":[
            {
                "iscrowd":0,
                "image_id":1,
                "bbox":[
                    0.0,
                    194.0,
                    189.0,
                    79.0
                ],
                "segmentation":[],
                "category_id":0,
                "id":1,
                "area":14931
            },
            {
                "iscrowd":0,
                "image_id":1,
                "bbox":[
                    283.0,
                    194.0,
                    247.0,
                    84.0
                ],
                "segmentation":[],
                "category_id":0,
                "id":2,
                "area":20748
            }
        ],
        "categories":[
            {
                "id":0,
                "name":"car",
                "supercategory":"car"
            }
        ]
    },  "data_folder/training_images/", tmppath, "logs/"]

def test_coco_to_yolo_conver(dummy_coco):
    json_data, imgpath, savejsonpath, logpath =dummy_coco
    CocoToYoloConverter(json_data, imgpath, savejsonpath, logpath)
    img_path = os.path.join(savejsonpath, "images")
    label_path = os.path.join(savejsonpath, "labels")
    assert len(os.listdir(img_path)) == len(os.listdir(label_path))
    assert len(os.listdir(savejsonpath)) == 5