import argparse
from typing import Dict
from deep_vision_tool.dataset_conversion.data_to_coco import CocoConverter
from deep_vision_tool.dataset_conversion.data_to_yolo import YOLOConverter
from deep_vision_tool.dataset_conversion.coco_to_yolo import CocoToYoloConverter
from deep_vision_tool.utils.file_utils import read_from_json


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_data_path", required=True, type=str, help="Provide the file path depending on the type of conversion you need to perform")
    parser.add_argument("--path_to_images", required=True, type=str, help="Provide the path to the image file")
    parser.add_argument("--storage_path", required=True, type=str, help="Pass the directory where we want to store the data")
    parser.add_argument("--log_directory", required=True, type=str, help="Enter the directory to put the log file")
    parser.add_argument("--type_of_conversion", required=True, type=str, help="Enter the type of conversion [to_coco, to_yolo, yolo_to_coco]")
    return parser.parse_args()

def validate_input_parameters(params):
    assert isinstance(params, dict), "Input should be a dictionary"
    
    expected_keys = {'json_data_path', 'path_to_images', 'storage_path', 'log_directory', 'type_of_conversion'}
    assert set(params.keys()) == expected_keys, f"Expected keys: {expected_keys}"

    assert isinstance(params['json_data_path'], str), "json_data_path should be a string"
    assert isinstance(params['path_to_images'], str), "path_to_images should be a string"
    assert isinstance(params['storage_path'], str), "storage_path should be a string"
    assert isinstance(params['log_directory'], str), "log_directory should be a string"
    assert isinstance(params['type_of_conversion'], str), "type_of_conversion should be a string"

    assert params['json_data_path'].endswith('.json'), "json_data_path should end with '.json'"
          
def check_parse_argument(args: Dict):
    """
        data_folder 
            | training_images
            | file.json
    """
    validate_input_parameters(args) # performing all the confirmation
    js_data = read_from_json(args["json_data_path"])
    if args["type_of_conversion"] == "to_coco":
        CocoConverter(json_data=js_data, path_to_image=args["path_to_images"], save_json_path=args["storage_path"],logger_output_dir=args["log_directory"])
    if args["type_of_conversion"] =="to_yolo":
        YOLOConverter(json_data=js_data, path_to_image=args["path_to_images"], save_json_path=args["storage_path"],logger_output_dir=args["log_directory"])
    if args["type_of_conversion"] =="yolo_to_coco":
        CocoToYoloConverter(json_data=js_data, path_to_image=args["path_to_images"], save_json_path=args["storage_path"],logger_output_dir=args["log_directory"])



if __name__ == "__main__":
    parsed_output = parse_opt()
    check_parse_argument(parsed_output.__dict__)

#python data_conversion_test.py --json_data_path "data_folder/file.json" --path_to_images "data_folder/training_images/" --storage_path "data_folder/" --log_directory "logs/" --type_of_conversion "to_coco"  