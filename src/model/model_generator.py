from river import stream
import numpy as np
import typing
from river import forest
from river import tree 
from river import metrics
from river import evaluate
import river
import logging
import os
import time

DEFAULT_N_MODELS = 10
DEFAULT_MAX_DEPTH = None
DEFAULT_MAX_SIZE = 100.0
DEFAULT_SEED = None

LOG_FILE_PATH = "/clusterlivenfs/lcap/ids-online/IDS_ONLINE_FILES/output/metrics.log"


AVAILABLE_ALGORITHMS = {
    "ARFClassifier": forest.ARFClassifier
    #TODO: Add more algorithms
}

AVAILABLE_METRICS = {
    "Accuracy": metrics.Accuracy,
    "Recall": metrics.Recall,
    "F1_Score": metrics.F1,
    "Precision": metrics.Precision,
    "ConfusionMatrix": metrics.ConfusionMatrix
}

class ModelGenerator():
    def __init__(self, config: typing.Dict, features_names):
        
        '''
        - features_names -> array with names for every feature
        '''        
        self._algorithm = config["config_model"]["algorithm"]
        self._features_names = features_names
        self._dataset_train = self.__build_dataset(config["dataset_train_load_paths"])
        self._dataset_test = self.__build_dataset(config["dataset_test_load_paths"])
        self._model = self.__build_algorithm(config["config_model"])
        self._metric = []
        self._start_time = 0
        self._number_samples = 0
        self._end_time = 0
        self._number_of_test_samples = self.__test_dataset_samples(config["dataset_test_load_paths"])
        self._time_per_sample = 0
        
        for metrics in config["config_model"]["metric"]:
            self._metric.append(AVAILABLE_METRICS[metrics]())
            
        self._logger = logging.getLogger(__name__)
        logging.basicConfig(
        filename=LOG_FILE_PATH,  
        filemode='a',         
        format='%(asctime)s - %(levelname)s - %(message)s', 
        level=logging.INFO)    

    def __iter(self, features, labels):
        
        try:
            self._dataset = stream.iter_array(features, labels, self._features_names)
        except:
            raise KeyError(f"Wrong iter_array parameter format!")
        
        return self._dataset
    
    def __test_dataset_samples(self, paths_dictionary: typing.Dict):
        features_array = np.load(paths_dictionary["X_path"])
        features_array = features_array.f.arr_0
        
        return features_array.shape[0]
    
    def __build_dataset(self, paths_dictionary: typing.Dict):
        
        features_array = np.load(paths_dictionary["X_path"])
        features_array = features_array.f.arr_0
        
        labels_array = np.load(paths_dictionary["y_path"])
        labels_array = labels_array.f.arr_0
        
        print(f"Built dataset with shape: {features_array.shape}")
            
        return self.__iter(features_array, labels_array)  
          
    def __build_algorithm(self, model_config: typing.Dict):
        
        n_models = model_config.get('n_models', DEFAULT_N_MODELS)
        max_depth = model_config.get('max_depth', DEFAULT_MAX_DEPTH)
        max_size = model_config.get('max_size', DEFAULT_MAX_SIZE)
        seed = model_config.get('seed', DEFAULT_SEED)
        # TODO: Add more hyperparameters
                
        return AVAILABLE_ALGORITHMS[self._algorithm](n_models=n_models, max_depth=max_depth, max_size=max_size, seed=seed)
        # return AVAILABLE_ALGORITHMS[self._algorithm]()
    
    def run(self, process):
        
        if process == "train":
            dataset = self._dataset_train
        elif process == "test":
            self._start_time = time.time()
            dataset = self._dataset_test
            
        for x, y in dataset:
            y_pred = self._model.predict_one(x)   
            self._model.learn_one(x, y)
            if y_pred is not None and process == "test":
                for metrics in self._metric:
                    metrics.update(y, y_pred)  
         
        if process == "test":        
            self._end_time = time.time()
            self._time_per_sample = self._number_of_test_samples / (self._end_time - self._start_time)
        

    def show_metric(self):
        for metrics in self._metric:
            print(metrics)
        
    def save_metric(self, config: typing.Dict):
        
        data_config = config["config"]
        model_config = config["config_model"]

        log = ""
         
        if model_config["train_and_test"]:
            
            if model_config["removed_from_train"] == False:
                log = f""" ALGORITHM: {model_config["algorithm"]} 
                                  DATA: {data_config["labeling_schema"]}_train + {data_config["labeling_schema"]}_test
                                  WINDOW SIZE: {data_config["window_size"]} 
                                  WINDOW SLIDE: {data_config["window_slide"]} 
                                  FEATURES SIZE: {model_config["feature_size"]} 
                                  AGGREGATION_METHOD: {data_config["aggregation_method"]} 
                                  NUMBER OF TREES: {model_config["n_models"]} 
                                  MAX_DEPTH: {model_config["max_depth"]} 
                                  SEED: {model_config["seed"]}
                                  REMOVED FROM DATASET: {data_config["remove_attack"]}
                                  FULL TEST DATASET PREDICTION TIME: {self._end_time - self._start_time} seconds
                                  PREDICTION TIME PER SAMPLE: {self._time_per_sample}"""
                                    
            else:
                log = f""" ALGORITHM: {model_config["algorithm"]} 
                                  DATA: {data_config["labeling_schema"]}_train + {data_config["labeling_schema"]}_test
                                  WINDOW SIZE: {data_config["window_size"]} 
                                  WINDOW SLIDE: {data_config["window_slide"]} 
                                  FEATURES SIZE: {model_config["feature_size"]} 
                                  AGGREGATION_METHOD: {data_config["aggregation_method"]} 
                                  NUMBER OF TREES: {model_config["n_models"]} 
                                  MAX_DEPTH: {model_config["max_depth"]} 
                                  SEED: {model_config["seed"]}
                                  REMOVED FROM TRAIN: {data_config["remove_attack"]}
                                  FULL TEST DATASET PREDICTION TIME: {self._end_time - self._start_time} seconds
                                  PREDICTION TIME PER SAMPLE: {self._time_per_sample}"""
        
        else:
            log = f""" ALGORITHM: {model_config["algorithm"]} 
                                  DATA: {data_config["labeling_schema"]}_{data_config["suffix"]} 
                                  WINDOW SIZE: {data_config["window_size"]} 
                                  WINDOW SLIDE: {data_config["window_slide"]} 
                                  FEATURES SIZE: {model_config["feature_size"]} 
                                  AGGREGATION_METHOD: {data_config["aggregation_method"]} 
                                  NUMBER OF TREES: {model_config["n_models"]} 
                                  MAX_DEPTH: {model_config["max_depth"]} 
                                  SEED: {model_config["seed"]}
                                  REMOVED FROM DATASET: {data_config["remove_attack"]}
                                  FULL TEST DATASET PREDICTION TIME: {self._end_time - self._start_time} seconds
                                  PREDICTION TIME PER SAMPLE: {self._time_per_sample}"""
                            
        log += "\n                                 "

        for i, metrics in enumerate(config['config_model']['metric']):
            
            if metrics == "ConfusionMatrix":
                #TODO: Implement confusion matrix
                continue
            
            log += f" | {metrics.upper()} {self._metric[i].get()}"
            
        log += " |"
        self._logger.info(log)
        