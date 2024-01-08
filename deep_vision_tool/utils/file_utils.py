from typing import Dict, List, Any, Tuple, Union
import numpy as np
import cv2 
import pickle
import json
import matplotlib.pyplot as plt
import os 
from PIL import Image

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1
font_color = (255, 255, 255)

def read_from_json(file_path: str) -> Any:
    """
    Read data from a JSON file.

    Args:
    - file_path (str): Path to the JSON file.

    Returns:
    - Any: Parsed data from the JSON file.
    """
    return json.load(open(file_path, "r"))


def save_to_json(file_path: str, results: Dict) -> None:
    """
    Save data to a JSON file.

    Args:
    - file_path (str): Path to the JSON file.
    - results (Dict): Data to be saved to the file.
    """
    
    with open(file_path, "w", encoding="utf8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=4)

def write_jsonl(filename, data_list):
    """
    Write a list of dictionaries to a JSON Lines (JSONL) file.

    Parameters:
    - filename (str): The name of the file to be written.
    - data_list (list): A list of dictionaries to be written to the file.

    Example:
    >>> data_list = [
    ...     {"name": "John", "age": 25},
    ...     {"name": "Alice", "age": 30},
    ...     {"name": "Bob", "age": 22}
    ... ]
    >>> write_jsonl("output.jsonl", data_list)
    """
    with open(filename, 'w') as file:
        for data in data_list:
            json_line = json.dumps(data)
            file.write(json_line + '\n')
    
def read_jsonl(filename):
    """
    Read a JSON Lines (JSONL) file and return a list of dictionaries.

    Parameters:
    - filename (str): The name of the JSONL file to be read.

    Returns:
    - list: A list of dictionaries containing the parsed JSON objects from the file.

    Example:
    >>> data_list = read_jsonl("input.jsonl")
    """
    data_list = []
    with open(filename, 'r') as file:
        for line in file:
            # Strip newline character and parse JSON
            json_data = json.loads(line.strip())
            data_list.append(json_data)
    return data_list

def read_from_image(img_path: str) -> Tuple[Union[np.array, int, int]]:
    """
    Read an image and return its numpy array along with width and height.

    Args:
    - img_path (str): Path to the image file.

    Returns:
    - Tuple[Union[np.array, int, int]]: Tuple containing the image array, height, and width.
    """
    img = Image.open(img_path)
    height = img.height
    width  = img.width
    return np.array(img), height,  width


def convert_bbox_to_coco_bbox(bbox: List, type:str="none") -> List:
    """
    Convert bounding box coordinates to COCO format.

    Args:
    - bbox (List): List containing bounding box coordinates [x1, y1, x2, y2].

    Returns:
    - List: Converted bounding box coordinates in COCO format [x, y, width, height].
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if type != "classification":
        assert w >= 0 and h >= 0, "Negative width or height"
        assert w != 0 or h != 0, "Zero width and height not allowed"
    return [int(x1), int(y1), int(w), int(h)]


def convert_bbox_to_yolo_bbox(bbox: List, is_coco: bool = False, type:str="object_detection") -> List:
    """
    Convert bounding box coordinates to YOLO format.

    Args:
    - bbox (List): List containing bounding box coordinates [x1, y1, x2, y2].
    - is_coco (bool): Flag indicating whether the input bbox is in COCO format.

    Returns:
    - List: Converted bounding box coordinates in YOLO format [x_center, y_center, width, height].
    """
    if not is_coco:
        x1, y1, x2, y2 = bbox
        x, y, w, h = convert_bbox_to_coco_bbox([x1, y1, x2, y2], type)
    else:
        x, y, w, h = bbox

    x_center = (x + (x + w)) / 2
    y_center = (y + (y + h)) / 2

    return [x_center, y_center, w, h]


def yolo_normalization(yolo_bbox: List, imgH: int, imgW: int) -> List:
    """
    Normalize YOLO bounding box coordinates.

    Args:
    - yolo_bbox (List): List containing YOLO format bounding box coordinates [x_center, y_center, width, height].
    - imgH (int): Image height.
    - imgW (int): Image width.

    Returns:
    - List: Normalized YOLO bounding box coordinates.
    """
    x_center, y_center, w, h = yolo_bbox

    x_center = x_center / imgW
    y_center = y_center / imgH

    w /= imgW
    h /= imgH
    return [x_center, y_center, w, h]


def get_all_categories(json_data: List[Dict[str, any]]) -> List[str]:
    """
    Get all unique categories from a list of JSON data.

    Args:
    - json_data (List[Dict[str, any]]): List of JSON records.

    Returns:
    - List[str]: List of unique category labels.
    """
    try:
        return list(sorted(set([annt["label"] for record in json_data for annt in record["annotations"]])))
    except json.JSONDecodeError as e:
        raise e


def read_text_file(filepath: str) -> str:
    """
    Read text content from a file.

    Args:
    - filepath (str): Path to the text file.

    Returns:
    - str: Text content.
    """
    return open(filepath, "r").read()


def write_to_text(filepath: str, records: str) -> None:
    """
    Write records to a text file.

    Args:
    - filepath (str): Path to the text file.
    - records (str): Records containing information of category, centerx, centery, x, y.
    """
    with open(filepath, "a", encoding="utf8") as fp:
        fp.write(f"{records}\n")


def is_dir_check(list_of_paths: List) -> None:
    """
    Check if directories exist, and create them if not.

    Args:
    - list_of_paths (List): List of paths to be checked.
    """
    for path in list_of_paths:
        if not os.path.isdir(path):
            os.mkdir(path)


def save_img(image_path: str, img: np.array) -> None:
    """
    Save an image to a specified path.

    Args:
    - image_path (str): Path to save the image.
    - img (np.array): Image in numpy array format.
    """
    cv2.imwrite(image_path, img)


def apply_bbox_to_img(allbboxes: List[List], img: np.array, labels: List[str]) -> np.array:
    """
    Apply bounding boxes to an image.

    Parameters:
    - allbboxes (List[List]): List of lists containing bounding box coordinates [x1, y1, width, height].
    - img (np.array): Image in numpy array format (w, h, channel).

    Returns:
    - np.array: Image with bounding boxes applied.

    Note:
    - Bounding box coordinates are assumed to be in the format [x1, y1, width, height].
    """

    assert len(allbboxes) == len(labels)
    for idx, bbox in enumerate(allbboxes):
        label = labels[idx]
        text =f"Label: {label}"
        x1, y1, w, h = list(map(int, bbox))
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_position = (x1 + (w - text_size[0]) // 2, y1 - 10)
        cv2.putText(img, text, text_position, font, font_scale, font_color, font_thickness)
        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h),  color=(0, 0, 255),  thickness=2)
    return img

def save_categories(categories: List, path_to_store_label: str) -> None:
    """
    Save categories to label-related files.

    Args:
    - categories (List): List of labels.
    - path_to_store_label (str): Path to store label-related files.
    """
    c2index = {label: idx for idx, label in enumerate(categories)}
    index2c = {value: key for key, value in c2index.items()}
    for filename in ["labels.txt", "label2index.json", "index2label.json"]:
        if filename.endswith("txt"):
            for cat in categories:
                write_to_text(os.path.join(path_to_store_label, filename), cat)
        else:
            # Store in JSON format 
            record = c2index if "label2index" in filename else index2c 
            save_to_json(os.path.join(path_to_store_label, filename), record)


def convert_yolo_to_coco(bboxs: List, width: int, height: int) -> List:
    """
    Convert YOLO bounding box coordinates to COCO format.

    Args:
    - bboxs (List): List of YOLO annotations.
    - width (int): Image width.
    - height (int): Image height.

    Returns:
    - List: Converted bounding box coordinates in COCO format [x, y, width, height].
    """
    assert len(bboxs) == 4
    bboxs = list(map(float, bboxs))
    x_center, y_center, w, h = bboxs
    bbox_x = (float(x_center) - float(w) / 2) * width
    bbox_y = (float(y_center) - float(h) / 2) * height
    w = float(w) * width
    h = float(h) * height
    return [bbox_x, bbox_y, w, h]


def bbox_to_segmentation(bbox: List) -> List:
    """
    Convert COCO bounding box to segmentation.

    Args:
    - bbox (List): List containing COCO bounding box coordinates [x, y, width, height].

    Returns:
    - List: List containing segmentation points [x1, y1, x2, y2, ..., xn, yn].
    """
    x, y, width, height = bbox

    # Calculate the four corner points of the bounding box
    x1, y1 = x, y
    x2, y2 = x + width, y
    x3, y3 = x + width, y + height
    x4, y4 = x, y + height

    # Return the segmentation as a list of x, y coordinates
    segmentation = [x1, y1, x2, y2, x3, y3, x4, y4]

    return segmentation


def is_coco_format(data: Dict[str, Any], type: str="object_detection") -> bool:
    """
    Check if the data is in COCO format.

    Args:
    - data (Dict[str, Any]): Data to be checked.

    Returns:
    - bool: True if the data is in COCO format, False otherwise.
    """
    # Check for the presence of required keys
    print(data)
    required_keys = {'annotations', 'images', 'categories'}
    if not all(key in data for key in required_keys):
        return False

    # Check the structure of the 'annotations' key
    if not isinstance(data['annotations'], list):
        return False

    for annotation in data['annotations']:
        # Check for the presence of required keys in each annotation
        annotation_keys = {'image_id', 'category_id', 'bbox'}
        if not all(key in annotation for key in annotation_keys):
            return False

        # Check the structure of the bounding box
        if type != "classification":
            if not isinstance(annotation['bbox'], list) or len(annotation['bbox']) != 4:
                return False
        if type == "classification":
            if not isinstance(annotation['bbox'], list) or len(annotation['bbox']) == 4:
                return False

    # Check the structure of the 'images' key
    if not isinstance(data['images'], list):
        return False

    # Check the structure of the 'categories' key
    if not isinstance(data['categories'], list):
        return False

    return True

def visualize_and_save(img: np.array, save_directory: str, filename: str) -> Union[str, bool]:
    """
    Display an image and save it to the specified directory.

    Parameters:
    - img (np.array): The image to visualize.
    - save_directory (str): The directory to save the image.
    - filename (str): The name of the file to save.

    Returns:
    - Union[str, bool]: Returns a string if save_directory is empty, otherwise returns True after saving the image.
    """
    # Save the image inside the directory
    if save_directory is None:
        return "Save directory is empty"
    else:
        # including the label name 
        img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()
        cv2.imwrite(os.path.join(save_directory, filename), img)
        return True

def visualize_segmentation_mask(coordinates_list: List[List[Union[int, float]]], img: np.array,
                                width: int, height: int, save_directory: str, filename: str, labels: List[str]):
    """
    Visualize segmentation masks on top of the original image and save the result.

    Parameters:
    - coordinates_list (List[List[Union[int, float]]]): List of segmentation points for a single image.
    - img (np.array): The original image.
    - width (int): Width of the image.
    - height (int): Height of the image.
    - save_directory (str): The directory to save the visualized image.
    - filename (str): The name of the file to save.

    Returns:
    - None: Displays the visualized image and saves it to the specified directory.
    """

    masks = []
    label_positions = dict()
    for idx, segment in enumerate(coordinates_list):
        text = f"Label: {labels[idx]}"
        label_position = (segment[0], segment[1] - 10)
        if label_positions.get(text) == None:
            label_positions[text] = [label_position]
        else:
            label_positions[text].append(label_position)
        segmentation = np.array(segment, dtype=np.int32).reshape((-1, 2))
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [segmentation], color=1)
        masks.append(mask)

    # Add these masks on top of the image
    overlay = img.copy()
    color: Tuple[int, int, int] = (255, 0, 0)
    color = np.asarray(color)

    for mask in masks:
        masked = np.ma.MaskedArray(img, mask=cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), fill_value=color)
        overlay = cv2.addWeighted(overlay, 1, masked.filled(), 0.5, 0)

    # Save the visualized image and display it
    if save_directory is not None:
        for label, positions in label_positions.items():
            for pos in positions:
                cv2.putText(img, label, pos, font, font_scale, font_color, font_thickness)
                cv2.putText(overlay, label, pos, font, font_scale, font_color, font_thickness)
        plt.subplot(1, 2, 1), plt.imshow(img), plt.title('Original Image')
        plt.subplot(1, 2, 2), plt.imshow(overlay), plt.title('Overlay')
        cv2.imwrite(os.path.join(save_directory, "seg_" + filename), cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.show()  

def store_pickle(data, file_path, filename):
    """
    Store data in a pickle file.

    Parameters:
    - data: Any data structure to be stored.
    - file_path (str): Path where the pickle file will be saved.

    Returns:
    - None
    """
    with open(os.path.join(file_path,filename), 'wb') as file:
        pickle.dump(data, file)
    print(f'Data successfully stored in {file_path}')

def read_pickle(file_path, filename="coco_img_obj.pickle"):
    """
    Read data from a pickle file.

    Parameters:
    - file_path (str): Path of the pickle file to be read.

    Returns:
    - Any: Data read from the pickle file.
    """
    filePath = file_path if filename == "" else os.path.join(file_path, filename)

    with open(filePath, 'rb') as file:
        data = pickle.load(file)
    print(f'Data successfully read from {file_path}')
    return data


def convert_to_classification_format(json_data):
    """
    Convert the given JSON data from object detection format to classification format.

    This function modifies the input JSON data by updating the "bbox" and "segmentation"
    fields for each annotation to be consistent with classification format, ensuring that
    each annotation has placeholder values for bounding box and segmentation.

    Parameters:
    - json_data (list): A list of dictionaries representing JSON data in object detection format.
                       Each dictionary should have an "annotations" field containing a list
                       of annotations, where each annotation includes a "bbox" and "segmentation" field.

    Returns:
    - list: The modified JSON data with "bbox" and "segmentation" fields updated for each annotation,
            now containing placeholder values.

    Example:
        input_json_data = [
            {"annotations": [{"bbox": [x1, y1, x2, y2], "segmentation": [s1, s2, ..., s8]}, ...]},
            {"annotations": [{"bbox": [x1, y1, x2, y2], "segmentation": [s1, s2, ..., s8]}, ...]},
            ...
        ]
    output_json_data = convert_to_classification_format(input_json_data)
    """
    for i in range(len(json_data)):
        temp_annt = []
        for annt in json_data[i]["annotations"]:
            annt["bbox"] = [0] * 4
            annt["segmentation"] = [0] * 8
            temp_annt.append(annt)
        
        json_data[i]["annotations"] = temp_annt
    return json_data

