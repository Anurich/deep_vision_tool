from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from deep_vision_tool.image_object.Image import ImageInfo
from typing import List, Dict
from deep_vision_tool.utils.file_utils import read_pickle, save_to_json,write_jsonl
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
import os 

class FeatureExtraction:
    def __init__(self, feature_extractor_model: AutoFeatureExtractor, image_object_path: str, output_path: str) -> None:
        allFolds = list(filter(lambda x: "Fold" in x, os.listdir(image_object_path)))
        if len(allFolds) == 0:
            allFolds = [""]
        self.image_processor = feature_extractor_model
        
        for fold in allFolds:
            train_file_path = os.path.join(image_object_path, fold, "train.pickle")
            dev_file_path   = os.path.join(image_object_path, fold, "dev.pickle")
            # data 
            train_data = read_pickle(train_file_path,"")
            dev_data   = read_pickle(dev_file_path,"")
            train_records = []
            dev_records   = []

            for imgObject in train_data:
                # since it is classification problem
                # we need to have only one label for each image 
                # let's check the annotations 
                label = list(set([annt.label for annt in imgObject.annotations]))
                assert len(label) == 1
                image_name = imgObject.filename
                image_path = imgObject.image_path
                img_file_path = os.path.join(image_path, image_name)
                images = Image.open(img_file_path)
                image_pixels = self.image_processor(images.convert("RGB"), return_tensors="pt")
                train_records.append({
                    "pixel_values": image_pixels["pixel_values"].numpy().tolist(),
                    "label": label[0]
                })
               

            for imgObject in dev_data:
                label = list(set([annt.label for annt in imgObject.annotations]))
                assert len(label) == 1

                image_name = imgObject.filename
                image_path = imgObject.image_path
                img_file_path = os.path.join(image_path, image_name)
                images = Image.open(img_file_path)
                image_pixels = self.image_processor(images.convert("RGB"),return_tensors="pt")
                dev_records.append({
                    "pixel_values": image_pixels["pixel_values"].numpy().tolist(),
                    "label": label[0]
                })
            # save these records to desired folder according to folds 
            # we need to save the train and test 
            train_data_storage = os.path.join(output_path, fold)
            dev_data_storage   = os.path.join(output_path, fold)
            os.makedirs(train_data_storage, exist_ok=True)
            os.makedirs(dev_data_storage, exist_ok=True)

            
            write_jsonl(os.path.join(train_data_storage, "train_data.json"), train_records)
            write_jsonl(os.path.join(dev_data_storage, "dev_data.json"), dev_records)

class ImageClassificationTransformer:
    def __init__(self, pretrained_model: str, image_object_path: str, output_path: str) -> None:
        self.pretrained_model =  pretrained_model
        self.auto_feature = AutoFeatureExtractor.from_pretrained(self.pretrained_model)
        self.model        = AutoModelForImageClassification.from_pretrained(self.pretrained_model)
        self.image_object_path = image_object_path
        # let's first Perform feature extraction
        FeatureExtraction(self.auto_feature, self.image_object_path, output_path)
        
