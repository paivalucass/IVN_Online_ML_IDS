import model_generator
import time
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Execute feature generation step')
    parser.add_argument('--config', required=True, help='JSON File containing the configs for the specified feature generation method')
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as config:
            config = json.load(config)

    except FileNotFoundError as e:
        print(f"parse_args: Error: {e}")
    except json.JSONDecodeError as e:
        print(f"parse_args: Error decoding JSON: {e}")
        
    # Create a features name array
    # features_names = np.array([f"feat_{i}" for i in range(config["config_model"]["feature_size"])]) # Alternative way
    
    features_names = []
    for i in range(config["config_model"]["feature_size"]):
        features_names.append(f"feat_{i}")
        
    # Load model    
    model = model_generator.ModelGenerator(config, features_names)
    
    if config["config_model"]["train_and_test"]:
        
        print("STARTING TRAINING AND TESTING")
        
        # Run model
        model.run("train")
        
        # Run model
        model.run("test")
        
        # Show output metric
        model.show_metric()
        
        # Save matrics in a log file
        model.save_metric(config)
        
    else:
        
        print("STARTING TESTING")

        # Run model
        model.run("test")
        
        # Show output metrics
        model.show_metric()
        
        # Save matrics in a log file
        model.save_metric(config)


if __name__ == "__main__":
    main()