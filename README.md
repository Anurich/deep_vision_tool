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
```bash
    |--data_folder
        |--yolo
            |--images
                |im.png
            |--labels
                |im.txt
            |labels.txt

```
* We need to pass the data into this structure if we want to use the yolo to coco conversion.
* In order to test it we can run the to_yolo, followed by yolo_to_coco.
# How To Use The Library.
```
    pip install -i https://test.pypi.org/simple/ deep-vision-tool==0.1.8
```
* After this we can simply import vision_tools. for example
```
    from deep_vision_tool.dataset_conversion.data_to_coco import CocoConverter
    from deep_vision_tool.dataset_conversion.data_to_yolo import YOLOConverter

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
    from deep_vision_tool.dataset_conversion.coco_to_yolo import CocoToYoloConverter

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
# Run Through ArgParser
* There are three category in which you can convert the data
* COCO category type (to_coco)
* YOLO category type (to_yolo)
* COCO2YOLO category type (coco_to_yolo)
* Dummy files are present inside the data_folder, if one needs to run it's own file just follow the structure of the files
```bash
    |--data_folder
    |     |--training_images
          |--file_name.json
```
* File_name.json should follow the structure of json mentioned above if we are converting to to_coco or to_yolo, the file structure is different. But, from COCO to YOLO we simply pass the coco structure.
```
python data_conversion_test.py --json_data_path "data_folder/file.json"
                               --path_to_images "data_folder/training_images/"
                               --storage_path "data_folder/coco"
                               --log_directory "logs/"
                               --type_of_conversion "to_coco"

python data_conversion_test.py --json_data_path "data_folder/file.json"
                               --path_to_images "data_folder/training_images/"
                               --storage_path "data_folder/yolo"
                               --log_directory "logs/"
                               --type_of_conversion "to_yolo"

python data_conversion_test.py --json_data_path "data_folder/coco/coco.json"
                               --path_to_images "data_folder/training_images/"
                               --storage_path "data_folder/coco"
                               --log_directory "logs/"
                               --type_of_conversion "coco_to_yolo"

python data_conversion_test.py --text_path "data_folder/yolo/labels/" \
                               --label_file_path "data_folder/yolo/labels.txt" \
                               --path_to_images "data_folder/yolo/images/" \
                               --storage_path "data_folder/yolo_to_coco/" \
                               --log_directory "logs/" \
                               --type_of_conversion "yolo_to_coco"


```


