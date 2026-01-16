import os
import sys
import argparse
from pathlib import Path
from server.config import FEATURES
from sklearn.model_selection import KFold
import numpy as np

# Ensure repo root is on sys.path so imports work when cwd != repo root
repo_root = Path(__file__).resolve().parent
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

try:
    import flwr as fl
except Exception as e:
    print("Error importing Flower (flwr). Make sure the 'flwr' package is installed in your Python environment.")
    raise

from shared.logger import setup_logger
logger = setup_logger(__name__)

# Defer heavy imports (model, data) until runtime to make startup errors clearer
Model = None
load_dataset = None


class FlwClient(fl.client.NumPyClient):
    def __init__(self, csv_path):
        self.csv_path = csv_path
        
        # 1. Update: load_dataset now returns 6 values (X_train, X_test, X_val, y_train, y_test, y_val)
        # You can specify keep_features/drop_features here if you want to experiment!
        X_train, X_test, X_val, y_train, y_test, y_val, full = load_dataset(
            self.csv_path,
            label_column="label",
            keep_features=FEATURES,
            drop_features=["hr_time_series", "resp_time_series", "resp_time_series"],
            test_size=0.2,
            val_size=0.1
        )
        
        # Save all partitions to the object (self)
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test

        self.X_cv = np.concatenate((X_train, X_val), axis=0)
        self.y_cv = np.concatenate((y_train, y_val), axis=0)

        # 2. Initialize Model
        input_size = self.X_train.shape[1]
        self.model = Model(input_size)
        
        # 3. Fit the scaler on training data immediately
        self.model.fit_scaler(self.X_train)
        logger.info(f"Initialized client. Features: {input_size}. Path: {csv_path}")

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):



        self.model.set_weights(parameters)
        
        # Get hyperparameters from the server config
        epochs = int(config.get("local_epochs", 5))
        batch_size = int(config.get("batch_size", 16))
        
        logger.info(f"Training on {len(self.X_train)} samples, validating on {len(self.X_val)}")

        # 4. Update: The new fit method takes X_train, y_train, X_val, y_val
        self.model.fit(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            epochs=epochs,
            batch_size=batch_size,
            leak_corr_threshold=0.999 # Or None if you want to skip leak check
        )
        
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
       
        self.model.set_weights(parameters)
        
        # 5. Update: Call your new regression-based evaluate function
        # It returns: huber, accuracy_threshold, mae
        huber, acc_threshold, mae = self.model.evaluate(self.X_test, self.y_test)
        
        logger.info(f"Eval: Huber={huber:.4f}, MAE={mae:.4f}, Acc(+/-5)={acc_threshold:.2%}")
        
        # 6. Return values that match your SaveModelStrategy
        return float(huber), len(self.X_test), {
            "accuracy": float(acc_threshold), 
            "mae": float(mae)
        }


def start_flower_client(server_address: str, csv_path: str = None, client_id: str = None):
    # 1. Resolve the CSV path if only client_id was provided
    if csv_path is None:
        if client_id is None:
            raise ValueError("Provide either --csv_path or --client_id")
        # Ensure we look in the 'dataset' folder relative to the repo root
        csv_path = str(repo_root.joinpath('dataset', f'group{client_id}_combined.csv'))

    # 2. Final safety check: does the file actually exist?
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {os.path.abspath(csv_path)}")

    # 3. Import Model and Data Loader (Lazy Import)
    global Model, load_dataset
    if Model is None or load_dataset is None:
            try:
                # 1. Import the module file explicitly
                import model.model as model_module
                # 2. Extract the class from that module
                Model = model_module.Model
                
                # 3. Repeat for data
                import data.load_data as data_module
                load_dataset = data_module.load_dataset
                
                print("--- [System] Successfully linked Model and Data Loader")
            except AttributeError as e:
                print(f"--- [ERROR] Found the file model.py, but 'Model' class is missing!")
                raise e
            except Exception as e:
                print(f"--- [ERROR] Could not find model/model.py: {e}")
                raise e

    # 4. Initialize the client (This triggers the loading and scaling)
    client = FlwClient(csv_path)
    
    # 5. Connect to the server
    logger.info(f"Connecting to Flower server at {server_address}...")
    fl.client.start_numpy_client(server_address=server_address, client=client)


if __name__ == "__main__":
    try:
        # 1. Immediate confirmation that the script is alive
        print("\n--- [System] client_flwr.py starting up ---")
        
        parser = argparse.ArgumentParser(description="Start a Flower federated client.")
        parser.add_argument("--server_address", type=str, required=True)
        parser.add_argument("--csv_path", type=str, required=False, default=None)
        parser.add_argument("--client_id", type=str, required=False, default=None)
        args = parser.parse_args()

        # 2. Logic for path resolution
        if args.csv_path:
            csv_path = args.csv_path
        elif args.client_id:
            # Using absolute path resolution to be safe on Windows
            csv_path = str(repo_root.joinpath('dataset', f'group{args.client_id}_combined.csv'))
        else:
            print("ERROR: You must provide either --csv_path or --client_id")
            sys.exit(1)

        print(f"--- [System] Target CSV: {csv_path}")
        
        # 3. Launch
        start_flower_client(
            server_address=args.server_address, 
            csv_path=csv_path, 
            client_id=args.client_id
        )

    except Exception as e:
        print("\n--- [CRITICAL ERROR] ---")
        import traceback
        traceback.print_exc()
        sys.exit(1)