from .coco_annotation import CocoAnnotation
from typing import List
class ImageInfo:
    def __init__(self,id: str, filename: str, image_path: str, im_width:int, im_height: int, annotations: List[CocoAnnotation]=None) -> None:
        self.id = id
        self.filename =filename
        self.im_width = im_width
        self.im_height = im_height
        self.annotations = annotations
        self.image_path = image_path
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, filename={self.filename}, image_path={self.image_path}, im_width={self.im_width}, im_height={self.im_height}, annotations={self.annotations})"