# Welcome To Vision Tool 
* This library help to easily convert data, to desired format **COCO, YOLO, COCO->YOLO, YOLO->COCO**
* Data must be pass to data converter in desired format as mentioned below.
* If we want to convert the data into **COCO** or **YOLO**, we need to pass the data in the format below.
```json
[
    {
        "image_id": 1,
        "img_name": "image1.jpg",
        "annotations": [
            {
                "label": labelname,
                "bbox":[x1, y1, x2, y2],
                "segmentation": [...],
                "area": area
            },
            {
                "label": labelname,
                "bbox": [x1, y1, x2, y2],
                "segmentation": [...],
                "area": area
            },
            // Add more annotations as needed
        ]
    }
]
```
*  If we want to convert the data from **COCO** to **YOLO**, In that case we need to pass the data into **COCO** format.
```json
"images":[
            {
                "height":380,
                "width":676,
                "id":1,
                "file_name":"vid_4_720.jpg"
            }
        ],
        "annotations":[
            {
                "iscrowd":0,
                "image_id":1,
                "bbox":[
                    0.0,
                    194.0,
                    189.0,
                    79.0
                ],
                "segmentation":[],
                "category_id":0,
                "id":1,
                "area":14931
            },
            {
                "iscrowd":0,
                "image_id":1,
                "bbox":[
                    283.0,
                    194.0,
                    247.0,
                    84.0
                ],
                "segmentation":[],
                "category_id":0,
                "id":2,
                "area":20748
            }
        ],
        "categories":[
            {
                "id":0,
                "name":"car",
                "supercategory":"car"
            }
        ]
```
* If we want to convert the data from **YOLO** to **COCO**, In that case we need to pass the data into format as shown below:
```
    Need to work
```
# How To Use The Library.
```
    pip install -i https://test.pypi.org/simple/ deep-vision-tool==0.1.8
```
* After this we can simply import vision_tools. for example
```
    from deep_vision_tool.dataset_conversion.data_to_coco import CocoConverter
    from deep_vision_tool.dataset_conversion.data_to_yolo import YOLOConverter
    from deep_vision_tool.dataset_conversion.coco_to_yolo import CocoToYoloConverter

    # let's understand how we can convert the data to either COCO or YOLO format
    js_data = [
        {
        'image_id': 1,
        'img_name': 'vid_4_720.jpg',
        'annotations': 
            [
                {'bbox': [0, 194, 189, 273], 'label': 'car'},
                {'bbox': [283, 194, 530, 278], 'label': 'car'}
            ],
        }
    ]
    image_path="images/"
    path_to_store_data = "data/"
    logs_dir ="logs/"
    CocoConverter(js_data, image_path, path_to_store, logs_dir)
    YOLOConverter(js_data, image_path, path_to_store, logs_dir)
```
* In this Example converting the COCO to YOLO format.
```
    # if we want to convert coco to yolo
    js_data = {
            "images":[
                {
                    "height":380,
                    "width":676,
                    "id":1,
                    "file_name":"vid_4_720.jpg"
                }
            ],
            "annotations":[
                {
                    "iscrowd":0,
                    "image_id":1,
                    "bbox":[
                        0.0,
                        194.0,
                        189.0,
                        79.0
                    ],
                    "segmentation":[],
                    "category_id":0,
                    "id":1,
                    "area":14931
                },
                {
                    "iscrowd":0,
                    "image_id":1,
                    "bbox":[
                        283.0,
                        194.0,
                        247.0,
                        84.0
                    ],
                    "segmentation":[],
                    "category_id":0,
                    "id":2,
                    "area":20748
                }
            ],
            "categories":[
                {
                    "id":0,
                    "name":"car",
                    "supercategory":"car"
                }
            ]
        }
    image_path="images/"
    path_to_store_data = "data/"
    logs_dir ="logs/"
    CocoToYoloConverter(js_data,image_path, path_to_store, logs_dir)
```
