from river import stream
import numpy as np
import typing
from river import forest
from river import tree 
from river import metrics
from river import evaluate

DEFAULT_N_MODELS = 10
DEFAULT_MAX_DEPTH = None
DEFAULT_MAX_SIZE = 100.0
DEFAULT_SEED = None

AVAILABLE_ALGORITHMS = {
    "ARFClassifier": forest.ARFClassifier
    # TODO: Add more algorithms
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
        
        print(features_array.shape)
        
        print(labels_array[0])
        
        return self.__iter(features_array, labels_array)  
          
    def __build_algorithm(self, model_config: typing.Dict):
        
        n_models = model_config.get('n_models', DEFAULT_N_MODELS)
        max_depth = model_config.get('max_depth', DEFAULT_MAX_DEPTH)
        max_size = model_config.get('max_size', DEFAULT_MAX_SIZE)
        seed = model_config.get('seed', DEFAULT_SEED)
        
        return AVAILABLE_ALGORITHMS[self._algorithm](n_models=n_models, max_depth=max_depth, max_size=max_size, seed=seed)
    
    def run(self):
        