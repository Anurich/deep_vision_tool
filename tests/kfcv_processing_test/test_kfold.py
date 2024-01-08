import pytest
from deep_vision_tool.image_object.annotation import Annotation
from deep_vision_tool.image_object.Image import ImageInfo
from deep_vision_tool.kfcv_processing.kfold import KFCV
import os 

# create the dummy data 
@pytest.fixture
def dummy_data():
    imgs = []
    for i in range(10):
        img_id = i
        annt_id = i+1
        annts = []
        for j in range(10):
            annts.append(Annotation(
                image_id=f"{img_id}",
                id=f"{annt_id}",
                bbox=[23 *j+1,21*j+1,22*j+1,12*j+1],
                segmentation=[],
                category_id=0,
                label="apple",
                type="object_detection"
            ))
        
        imgs.append(ImageInfo(
            id=img_id,
            filename="abc.png",
            image_path="/content/drive/",
            im_width=224,
            im_height=224,
            annotations=annts
        ))
    return imgs
    
def test_kfcv(dummy_data, tmp_path):
    log_path = tmp_path / "logs"
    output_path = tmp_path / "folds"
    log_path.mkdir()
    output_path.mkdir()

    kfcv = KFCV(output_path=output_path, data_object= dummy_data,log_dir=log_path)
    kfcv.kfold_cv()
    allFolds = list(filter(lambda x: "Fold_" in x, os.listdir(output_path)))

    assert len(allFolds) == 5
    for fold in allFolds:
        path = os.path.join(output_path, fold)
        assert len(os.listdir(path)) == 2
        assert "dev.pickle" in os.listdir(path)
        assert "train.pickle" in os.listdir(path)