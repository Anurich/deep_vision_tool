from transformers import AutoFeatureExtractor, AutoModelForImageClassification

class FeatureExtraction:
    def __init__(self) -> None:
        pass

class ImageClassificationTransformer:
    def __init__(self, pretrained_model: str) -> None:
        self.pretrained_model =  pretrained_model
        self.auto_feature = AutoFeatureExtractor.from_pretrained(self.pretrained_model)
        self.model        = AutoModelForImageClassification.from_pretrained(self.pretrained_model)

ImageClassificationTransformer("google/vit-base-patch16-224")
