from typing import Dict, List, Any, Tuple, Union
import numpy as np
import cv2 
import json
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
        json.dump(file_path, fp, ensure_ascii=False, indent=4)


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

def apply_bbox_to_img(annotations: List[Dict[str, any]], img: np.array)-> np.array:
    allbboxes = [annt["bbox"] for annt in annotations]
    for bbox in allbboxes:
        x1, y1, w, h = list(map(int, bbox))
        cv2.rectangle(img, (x1, y1), (x1+w, y1+h),  color=(0,0,255),  thickness=2)
    return img 
