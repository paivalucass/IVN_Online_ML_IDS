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
        self._dataset = self.__build_dataset(config["dataset_load_paths"])
        self._model = self.__build_algorithm(config["config_model"])
        self._metric = []
        
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
    
    def run(self):

        for x, y in self._dataset:
            y_pred = self._model.predict_one(x) 
            self._model.learn_one(x, y)
            if y_pred is not None:
                for metrics in self._metric:
                    metrics.update(y, y_pred)  
            print(f"Predicted:{y_pred} / Real:{y}")

    def show_metric(self):
        for metrics in self._metric:
            print(metrics)
        
    def save_metric(self, config: typing.Dict):
        
        data_config = config["config"]
        model_config = config["config_model"]
        
        log = f""" ALGORITHM: {model_config["algorithm"]} 
                                  DATA: {data_config["labeling_schema"]}_{data_config["suffix"]} 
                                  WINDOW SIZE: {data_config["window_size"]} 
                                  WINDOW SLIDE: {data_config["window_slide"]} 
                                  FEATURES SIZE: {model_config["feature_size"]} 
                                  AGGREGATION_METHOD: {data_config["aggregation_method"]} 
                                  NUMBER OF TREES: {model_config["n_models"]} 
                                  MAX_DEPTH: {model_config["max_depth"]} 
                                  SEED: {model_config["seed"]}
                                  REMOVED ATTACK: {data_config["remove_attack"]}"""
                        
        log += "\n                                 "

        for i, metrics in enumerate(config['config_model']['metric']):
            
            if metrics == "ConfusionMatrix":
                #TODO: Implement confusion matrix
                continue
            
            log += f" | {metrics.upper()} {self._metric[i].get()}"
            
        log += " |"
        self._logger.info(log)
        