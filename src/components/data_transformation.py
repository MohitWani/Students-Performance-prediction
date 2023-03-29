import os 
import sys 

import pandas as pd
import numpy as np
from dataclasses import dataclass
 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationconfig:
    processor_obj_path = os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationconfig()
    