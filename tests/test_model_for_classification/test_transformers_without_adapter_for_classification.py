import pytest
import os 
from deep_vision_tool.models_training.classification.huggingface_transformers_image_classification import ImageClassificationTransformer
from deep_vision_tool.dataset_conversion.data_to_coco import CocoConverter
from deep_vision_tool.dataset_conversion.data_to_yolo import YOLOConverter
from deep_vision_tool.kfcv_processing.kfold import KFCV
from deep_vision_tool.image_preprocessing.create_data_object import ObjectDetection
from deep_vision_tool.utils.file_utils import read_pickle

@pytest.fixture
def dummy_data():
    # we need to create a dataset 

    data = [
        {
            "image_id": 1,
            "img_name": "vid_4_720.jpg",
            "annotations":[
                {
                    "label": "car",
                    "bbox": [],
                    "segmentation": [],
                    "area":0
                },
                {
                    "label": "car",
                    "bbox": [],
                    "segmentation":[],
                    "area":0
                }
            ]
        },
        {
            "image_id": 2,
            "img_name": "aeroplane.jpg",
            "annotations":[
                {
                    "label": "aeroplane",
                    "bbox": [],
                    "segmentation": [],
                    "area":0
                },
                {
                    "label": "aeroplane",
                    "bbox": [],
                    "segmentation":[],
                    "area":0
                }
            ]
        }
    ]
    
    return data 
def test_transformers_classification_training_coco(dummy_data, tmp_path):
    ## coco conversion
    CocoConverter(json_data=dummy_data, 
                  path_to_image="data_folder/training_images/", 
                  save_json_path= tmp_path, 
                  logger_output_dir="logs",
                  type="classification")
    
    # we need to convert to the image object 
    tmp_image_object_storage = tmp_path / "image_object"
    tmp_image_object_storage.mkdir()
    image_object = ObjectDetection(filepath=tmp_path, 
                    type_of_data="coco",
                    save_image_object_path=tmp_image_object_storage,
                    image_path="data_folder/training_images/",
                    log_dir="logs",
                    type="classification"
                    )
    image_object.get_coco_image_object()
     ## perform kfold 
    tmp_path_folds = tmp_path / "folds"
    tmp_path_folds.mkdir()
    # let's read the image data object 
    kfold = KFCV(output_path=tmp_path_folds, data_object=image_object.coco_image_obj, n_splits=2, log_dir="logs")
    kfold.kfold_cv()
    features_path = tmp_path / "features"
    ImageClassificationTransformer("google/vit-base-patch16-224",image_object_path=tmp_path_folds, output_path=features_path)
    allFolds = os.listdir(features_path)
    assert len(allFolds) == 2
    for fold in allFolds:
        allFiles = os.listdir(os.path.join(features_path, fold))
        assert len(allFiles) == 2
        assert "dev_data.json" in allFiles
        assert "train_data.json" in allFiles
    

def test_transformers_classification_training_yolo(dummy_data, tmp_path):
    ## coco conversion
    tmp_path_yolo = tmp_path / "yolo"
    tmp_path_yolo.mkdir()
    YOLOConverter(json_data=dummy_data, 
                  path_to_image="data_folder/training_images/", 
                  save_json_path= tmp_path_yolo, 
                  logger_output_dir="logs",
                  type="classification")
    
    #we need to convert to the image object 
    tmp_image_object_storage = tmp_path / "image_object_yolo"
    tmp_image_object_storage.mkdir()
    image_object = ObjectDetection(filepath=tmp_path_yolo, 
                    type_of_data="yolo",
                    save_image_object_path=tmp_image_object_storage,
                    image_path="data_folder/training_images/",
                    log_dir="logs",
                    type="classification"
                    )
    image_object.get_yolo_image_object()



     ## perform kfold 
    tmp_path_folds = tmp_path / "folds_yolo"
    tmp_path_folds.mkdir()
    # let's read the image data object 
    kfold = KFCV(output_path=tmp_path_folds, data_object=image_object.yolo_image_obj, n_splits=2, log_dir="logs")
    kfold.kfold_cv()
    features_path = tmp_path / "features_yolo"
    ImageClassificationTransformer("google/vit-base-patch16-224",image_object_path=tmp_path_folds, output_path=features_path)
    allFolds = os.listdir(features_path)
    assert len(allFolds) == 2
    for fold in allFolds:
        allFiles = os.listdir(os.path.join(features_path, fold))
        assert len(allFiles) == 2
        assert "dev_data.json" in allFiles
        assert "train_data.json" in allFiles
    