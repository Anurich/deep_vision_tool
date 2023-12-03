import argparse
from typing import Dict


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_data_path", required=True, type=str, help="Provide the file path depending on the type of conversion you need to perform")
    parser.add_argument("--path_to_images", required=True, type=str, help="Provide the path to the image file")
    parser.add_argument("--storage_path", required=True, type=str, help="Pass the directory where we want to store the data")
    parser.add_argument("--log_directory", required=True, type=str, help="Enter the directory to put the log file")
    parser.add_argument("--type_of_conversion", required=True, type=str, help="Enter the type of conversion [to_coco, to_yolo, yolo_to_coco]")
    return parser.parse_args()
          
def check_parse_argument(dict_data: Dict):

if __name__ == "__main__":
    parsed_output = parse_opt()
    check_parse_argument(parsed_output.__dict__)
