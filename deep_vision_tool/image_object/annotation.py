from typing import List, Dict, Any, Union

class Annotation:
    def __init__(self, image_id: str, id: str, bbox: List[Union[int, float]], \
                  segmentation: List[Union[int, float]], category_id: int, label: str, type:str) -> None:
        self.image_id = image_id
        self.id = id
        self.bbox = bbox
        self.segmenation = segmentation
        self.category_id = category_id
        self.label = label
        self.type  = type

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(image_id={self.image_id},id={self.id}, bbox={self.bbox},segmentation={self.segmenation},category_id={self.category_id}, label={self.label}, type={self.type})"
    

    