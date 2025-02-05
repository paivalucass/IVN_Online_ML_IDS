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
import pickle
import enum
from sklearn.metrics import classification_report
from sklearn.ensemble import IsolationForest, RandomForestClassifier, HistGradientBoostingClassifier

DEFAULT_N_MODELS = 10
DEFAULT_MAX_DEPTH = None
DEFAULT_MAX_SIZE = 100.0
DEFAULT_SEED = None

river_models = {"ARFClassifier", "HATClassifier", "HTClassifier"}
sklearn_models = {"RandomForest"}

LOG_FILE_PATH = "/clusterlivenfs/lcap/ids-online/IDS_ONLINE_FILES/output/metrics.log"

AVAILABLE_ALGORITHMS = {
    "ARFClassifier": forest.ARFClassifier,
    "RandomForest": RandomForestClassifier,
    "HATClassifier": tree.HoeffdingAdaptiveTreeClassifier,
    "HTClassifier": tree.HoeffdingTreeClassifier
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
        self._framework = None
        
        if self._algorithm in sklearn_models:
            self._framework = "sklearn"
            self._features_array_train, self._labels_array_train = self.__build_dataset(config["dataset_train_load_paths"])
            self._features_array_test, self._labels_array_test = self.__build_dataset(config["dataset_test_load_paths"])
            self._report = None
            
        elif self._algorithm in river_models:
            self._framework = "river"
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
        
        if self._framework == "sklearn":
            return features_array, labels_array
        elif self._framework == "river":
            return self.__iter(features_array, labels_array)

        return None
          
    def __build_algorithm(self, model_config: typing.Dict):
        
        n_models = model_config.get('n_models', DEFAULT_N_MODELS)
        max_depth = model_config.get('max_depth', DEFAULT_MAX_DEPTH)
        max_size = model_config.get('max_size', DEFAULT_MAX_SIZE)
        seed = model_config.get('seed', DEFAULT_SEED)
        # TODO: Add more hyperparameters

        if self._framework == "river":
                    
            return AVAILABLE_ALGORITHMS[self._algorithm](n_models=n_models, max_depth=max_depth, max_size=max_size, seed=seed)
        
        elif self._framework == "sklearn":
            
            return AVAILABLE_ALGORITHMS[self._algorithm](n_estimators=n_models, max_depth=max_depth)

        return None
    
    def run(self, process):
        
        if self._framework == "river":
            
            if process == "train":
                dataset = self._dataset_train
                for x, y in dataset:
                    y_pred = self._model.predict_one(x)   
                    self._model.learn_one(x, y)
                    # if y_pred is not None and process == "test":
                    #     for metrics in self._metric:
                    #         metrics.update(y, y_pred)
                              
            elif process == "test":
                dataset = self._dataset_test
                self._start_time = time.time()
            
                for x, y in dataset:
                    y_pred = self._model.predict_one(x)   
                    # self._model.learn_one(x, y)
                    if y_pred is not None and process == "test":
                        for metrics in self._metric:
                            metrics.update(y, y_pred)
                              
            
                
        elif self._framework == "sklearn":
            
            if process == "train":
                features, labels = self._features_array_train, self._labels_array_train
                self._model.fit(features, labels)
            elif process == "test":
                features, labels = self._features_array_test, self._labels_array_test
                self._start_time = time.time()            
                prediction = self._model.predict(features)
                self._report = classification_report(labels, prediction, output_dict=True)
            
        if process == "test":        
                self._end_time = time.time()
                self._time_per_sample = (self._end_time - self._start_time) / self._number_of_test_samples
                
        
        return self._model
        
    def show_metric(self):
        
        if self._framework == "river":
            for metrics in self._metric:
                print(metrics)
                
        elif self._framework == "sklearn":
            print(f"Precision for class 0: {self._report['0']['precision']}")
            print(f"Recall for class 0: {self._report['0']['recall']}")
            print(f"F1-Score for class 0: {self._report['0']['f1-score']}")
            print(f"Support for class 0: {self._report['0']['support']}")

            print(f"Precision for class 1: {self._report['1']['precision']}")
            print(f"Recall for class 1: {self._report['1']['recall']}")
            print(f"F1-Score for class 1: {self._report['1']['f1-score']}")
            print(f"Support for class 1: {self._report['1']['support']}")

            # Overall accuracy
            print(f"Overall accuracy: {self._report['accuracy']}")
        
    def save_metric(self, config: typing.Dict):
        
        data_config = config["config"]
        model_config = config["config_model"]
        
        elapsed_time = self._end_time - self._start_time

        log = ""
        
        if self._framework == "river":
         
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
                                    FULL TEST DATASET PREDICTION TIME: {elapsed_time} Seconds
                                    PREDICTION TIME PER SAMPLE: {self._time_per_sample} Seconds
                                    NUMBER OF TEST SAMPLES: {self._number_of_test_samples}"""
                                        
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
                                    FULL TEST DATASET PREDICTION TIME: {elapsed_time} Seconds
                                    PREDICTION TIME PER SAMPLE: {self._time_per_sample} Seconds
                                    NUMBER OF TEST SAMPLES: {self._number_of_test_samples}"""
            
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
                                    FULL TEST DATASET PREDICTION TIME: {elapsed_time} Seconds
                                    PREDICTION TIME PER SAMPLE: {self._time_per_sample} Seconds
                                    NUMBER OF TEST SAMPLES: {self._number_of_test_samples}"""
                                
            log += "\n                                 "

            for i, metrics in enumerate(config['config_model']['metric']):
                
                if metrics == "ConfusionMatrix":
                    #TODO: Implement confusion matrix
                    continue
                
                log += f" | {metrics.upper()} {self._metric[i].get()}"
                
            log += " |"
            self._logger.info(log)
    

        elif self._framework == "sklearn":
            
            log = f""" ALGORITHM: {model_config["algorithm"]} 
                                DATA: {data_config["labeling_schema"]}_train + {data_config["labeling_schema"]}_test
                                WINDOW SIZE: {data_config["window_size"]} 
                                WINDOW SLIDE: {data_config["window_slide"]} 
                                FEATURES SIZE: {model_config["feature_size"]} 
                                AGGREGATION_METHOD: {data_config["aggregation_method"]} 
                                NUMBER OF ESTIMATORS: {model_config["n_models"]} 
                                MAX_DEPTH: {model_config["max_depth"]} 
                                REMOVED FROM DATASET: {data_config["remove_attack"]}
                                FULL TEST DATASET PREDICTION TIME: {elapsed_time} Seconds
                                PREDICTION TIME PER SAMPLE: {self._time_per_sample} Seconds
                                NUMBER OF TEST SAMPLES: {self._number_of_test_samples}
                                | GENERAL ACCURACY {self._report['accuracy']} | PRECISION NORMAL {self._report['0']['precision']} | RECALL NORMAL {self._report['0']['recall']} | F1-SCORE NORMAL {self._report['0']['f1-score']} | PRECISION ANOMALY {self._report['1']['precision']} | RECALL ANOMALY {self._report['1']['recall']} | F1-SCORE ANOMALY {self._report['1']['f1-score']} |"""
                                
            self._logger.info(log)