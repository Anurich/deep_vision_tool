from typing import List, Dict, Any, Union
class YoloAnnotation:
    def __init__(self, category_id: int, bbox: List[Union[int, float]]) -> None:
        
        self.category_id = category_id
        self.bbox = bbox

    def __repr__(self) -> str:
         return f"{self.__class__.__name__}(category_id={self.category_id}, bbox={self.bbox})"