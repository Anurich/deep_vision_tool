from typing import Dict, List, Any, Tuple, Union
import numpy as np
import cv2 
import json
import os 
from PIL import Image


def read_from_json(file_path: str):
    """
    Args
        input: file_path of type string
    Returns
        output: 
    
    """
    return  json.load(open(file_path, "r"))


def save_to_json(file_path: str, results):
    """
    Args
        file_path (str)
        results (Dictionary)
    """
    with open(file_path, "w", encoding="utf8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=4)


def read_from_image(img_path: str) -> Tuple[Union[np.array, int, int]]:
    """
    Args 
        img_path (str)
    Returns
        numpy array or image matrix
    """
    img = Image.open(img_path)
    height = img.height
    width  = img.width
    return img, height,  width


def convert_bbox_to_coco_bbox(bbox: List) -> List:
    """
        Args
            bbox is of type List 
        Returns
            converted bbox to coco format
    """
    x1, y1, x2, y2 = bbox
    w = x2-x1
    h = y2-y1
    assert w >= 0 and h >= 0, "Negative width or height"
    assert w != 0 or h != 0, "Zero width and height not allowed"
    return [int(x1), int(y1), int(w), int(h)]

def convert_bbox_to_yolo_bbox(bbox: List, is_coco:bool=False) -> List:
    """
        Args
            bbox is of type List 
        Returns
            converted bbox to yolo format
    """
    if is_coco==False:
        x1, y1, x2, y2 = bbox
        x, y, w, h = convert_bbox_to_coco_bbox([x1,y1,x2,y2])
    else:
        x, y, w, h = bbox
        
    x_center = (x+(x+w))/2
    y_center = (y+(y+h))/2

    return [x, y, x_center, y_center]

def yolo_normalization(yolo_bbox: List, imgH: int, imgW: int) -> List:

    """
    Args
        yolo_bbox with [x, y, x_center, y_center]
    Returns 
        List of normalized bbox 
    
    """
    x, y, x_center, y_center = yolo_bbox
    # let's normalized it 
    x_normalized  = x/imgW
    y_normalized  = y/imgH

    x_center /= imgW
    y_center /= imgH
    return [x_center, y_center,x_normalized, y_normalized] 


def get_all_categories(json_data: List[Dict[str, any]]) -> List[str]:
    """
        Args
            json_path: str 
        Returns
            List[str]: return the list of categories
    """
    try:
        return list(sorted(set([annt["label"] for record in json_data for annt in record["annotations"]])))
    except json.JSONDecodeError as e:
        raise e

def read_text_file(filepath: str):
    return open(filepath, "r").read()
    
def write_to_text(filepath: str, records: str):
    """
     Args:
        filepath: file location to save the text file
        records: it's the record containing the information of category, centerx, centery, x, y
    """
    
    with open(filepath, "a", encoding="utf8") as fp:
        fp.write(f"{records}\n")

def is_dir_check(list_of_paths: List)-> None:
    """
        Args: 
            List of path which need to be created if not exist
    """
    for path in list_of_paths:
        if not os.path.isdir(path):
            os.mkdir(path)

def save_img(image_path: str, img: np.array):
    """
     Args:
        image_path: path to image  location
        save_img_path: path where we want to save the image
    """
    cv2.imwrite(image_path, img)

def apply_bbox_to_img(annotations: List[Dict[str, any]], img: np.array)-> np.array:
    """
        Args:
            annotations : List of dictionary containing the information about bbox
            img: image of numpy array (w, h, channel)
    """
    allbboxes = [annt["bbox"] for annt in annotations]
    for bbox in allbboxes:
        x1, y1, w, h = list(map(int, bbox))
        cv2.rectangle(img, (x1, y1), (x1+w, y1+h),  color=(0,0,255),  thickness=2)
    return img 

def save_categories(categories: List, path_to_store_label: str):
    """
        Args: 
            categories: list of labels
    """
    c2index = {label:idx for idx, label in enumerate(categories)}
    index2c = {value: key for key, value in c2index.items()}
    for filename in ["labels.txt", "label2index.json", "index2label.json"]:
        if filename.endswith("txt"):
            for cat in categories:
                write_to_text(os.path.join(path_to_store_label, filename), cat)
        else:
            # let's store in json format 
            record = c2index if "label2index" in filename else index2c 
            save_to_json(os.path.join(path_to_store_label, filename),record)
            

