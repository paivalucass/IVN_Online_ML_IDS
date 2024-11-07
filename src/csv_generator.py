import pandas as pd
import numpy as np
import time
from scapy.all import *
import features_generator


config = {
    "dataset":"TOW_IDS_dataset",
    "labeling_schema":"TOW_IDS_dataset_multi_class",
    "multiclass":True,
}

generator_class = features_generator.CNNIDSFeatureGenerator(config=config)
generator_class.generate_features()

