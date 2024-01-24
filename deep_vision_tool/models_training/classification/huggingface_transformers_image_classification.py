from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from deep_vision_tool.image_object.Image import ImageInfo
from typing import List, Dict, Any
from deep_vision_tool.utils.file_utils import read_pickle, write_jsonl,is_dir_check
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


def _custom_transform_function_of_pretrained_model(image_processor):
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
        max_size = None
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)
        max_size = image_processor.size.get("longest_edge")

    train_transforms = Compose(
            [
                RandomResizedCrop(crop_size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )

    val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(crop_size),
                ToTensor(),
                normalize,
            ]
        )
    return train_transforms, val_transforms

class FeatureExtraction:
    def __init__(self, feature_extractor_model: AutoFeatureExtractor, 
                 image_object_path: str, output_path: str, train_transform:Any=None,
                 dev_transform: Any=None) -> None:
        

        allFolds = list(filter(lambda x: "Fold" in x, os.listdir(image_object_path)))
        if len(allFolds) == 0:
            allFolds = [""]
        self.image_processor = feature_extractor_model
        # for training
        self._processing(allFolds, image_object_path, 
                         "train.pickle", 
                         "train_data.json", 
                         output_path,
                         train_transform
                         )
        # for developement
        self._processing(allFolds, image_object_path, 
                         "dev.pickle", 
                         "dev_data.json", 
                         output_path,
                         train_transform
                         )
        

        
           
    def _processing(self, allFolds: Any, image_object_path: str,
                     file_path: str, storage_file: str, output_path: str, transfomrs: Any):
        for fold in allFolds:
            object_file_path = os.path.join(image_object_path, fold, file_path)
            # data 
            data = read_pickle(object_file_path,"")
            records = []

            for imgObject in data:
                # since it is classification problem
                # we need to have only one label for each image 
                # let's check the annotations 
                label = list(set([annt.label for annt in imgObject.annotations]))
                assert len(label) == 1
                image_name = imgObject.filename
                image_path = imgObject.image_path
                img_file_path = os.path.join(image_path, image_name)
                images = Image.open(img_file_path)
                if transfomrs == None:
                    image_pixels = self.image_processor(images.convert("RGB"), return_tensors="pt")
                else:
                    image_pixels =dict()
                    image_pixels["pixel_values"] = transfomrs(images.convert("RGB"))

                records.append({
                    "pixel_values": image_pixels["pixel_values"].numpy().tolist(),
                    "label": label[0]
                })
               
                # save these records to desired folder according to folds 
                # we need to save the train and test 
                data_storage = os.path.join(output_path, fold)
                is_dir_check([data_storage])
                write_jsonl(os.path.join(data_storage, storage_file), records)

class ImageClassificationTransformer:
    def __init__(self, pretrained_model: str, 
                 image_object_path: str, 
                 output_path: str, 
                 use_custom_transform_function_of_pretrained_mode:bool=False) -> None:
        self.auto_feature = AutoFeatureExtractor.from_pretrained(pretrained_model)
        self.model        = AutoModelForImageClassification.from_pretrained(pretrained_model)
        self.image_object_path = image_object_path
        # let's first Perform feature extraction
        train_transform, dev_transform = None, None
        if use_custom_transform_function_of_pretrained_mode:
            train_transform, dev_transform = _custom_transform_function_of_pretrained_model(self.auto_feature)
        
        FeatureExtraction(self.auto_feature, 
                          self.image_object_path, 
                          output_path,
                          train_transform,
                          dev_transform
                          )
        
