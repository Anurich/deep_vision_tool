from conftest  import *
from deep_vision_tool.image_preprocessing.object_detection import ObjectDetection

objimage = ObjectDetection("data_folder/coco/","coco", "data_folder/training_images/","logs")
imgObject = objimage.coco_postprocessing()
    
print(imgObject)
