# Welcome To Vision Tool 
* This library help to easily convert data, to desired format ["YOLO", "COCO"]
* Data must be pass to data converter in this format, before converting to YOLO or COCO.

```json

[
    {
        "image_id": 1,
        "img_name": "image1.jpg",
        "annotations": [
            {
                "label": labelname,
                "bbox": [x1, y1, x2, y2],
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