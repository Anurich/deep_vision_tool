from abc import ABC, abstractmethod
from typing import List, Dict

class Dataset(ABC):
    @abstractmethod
    def __init__(self, json_data: List[Dict[str, any]], path_to_image: str, 
                save_json_path: str, logger_output_dir: str, type_of_data_converstion: str) -> None:
        self.json_data = json_data
        self.path_to_image =path_to_image
        self.save_json_path = save_json_path
        self.logger_output_dir = logger_output_dir
        self.type_of_data_converstion=type_of_data_converstion

    @abstractmethod
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path_to_json = {self.path_to_json}, path_to_image={self.path_to_image}, save_json_path={self.save_json_path}, logger_output_dir={self.logger_output_dir}, type_of_data_converstion={self.type_of_data_converstion})"
