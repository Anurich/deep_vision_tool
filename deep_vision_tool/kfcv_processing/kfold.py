from sklearn.model_selection import KFold, StratifiedKFold
from ..image_object.annotation import Annotation
from ..image_object.Image import ImageInfo
from typing import List, Dict
from ..utils.logging_util import initialize_logging
import logging
import numpy as np
from utils.file_utils import store_pickle
import os 

class KFCV:
    def __init__(self, output_path: str,\
                 data_object: List[ImageInfo],log_dir: str, n_splits: int =5, random_state: int=None, \
                    shuffle: bool = False) -> None:
        """
            Initialize the cross-validator for splitting the data.
            Parameters:
            - output_path (str): The path to store the cross-validation results.
            - data_object (List[ImageInfo]): A list of ImageInfo objects representing the dataset.
            - n_splits (int, optional): Number of splits. Defaults to 5.
            - random_state (int, optional): Seed for the random number generator. Defaults to 42.
            - shuffle (bool, optional): Whether to shuffle the data before splitting. Defaults to False.

            Attributes:
            - output_path (str): The path to store the cross-validation results.
            - data_object (List[ImageInfo]): A list of ImageInfo objects representing the dataset.
            - n_splits (int): Number of splits.
            - random_state (int): Seed for the random number generator.
            - shuffle (bool): Whether to shuffle the data before splitting.
        """
        self.output_path = output_path
        self.data_object = data_object
        self.n_splits = n_splits
        self.shuffle= shuffle
        self.random_state = random_state
        self.logger = initialize_logging(log_dir)
    
    def kfold_cv(self):
        kfold = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        if self.random_state != None and self.shuffle==False:
            self.logger.error("Shuffle cannot be false, if random state is not None !!")
        else:
            for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(self.data_object)):
                train_data = np.array(self.data_object)[train_idx]
                test_data  = np.array(self.data_object)[test_idx]
                foldname = f"Fold_{fold_idx}"
                dev_file_path = os.path.join(self.output_path, foldname)
                train_file_path = os.path.join(self.output_path, foldname)
                os.makedirs(dev_file_path, exist_ok=True)
                os.makedirs(train_file_path, exist_ok=True)
                # now we can save these data into these files 
                store_pickle(train_data,train_file_path, "dev.pickle")
                store_pickle(test_data, dev_file_path, "train.pickle")