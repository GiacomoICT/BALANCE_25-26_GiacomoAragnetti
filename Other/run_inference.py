from pathlib import Path
import sys

# Identify the root directory (the parent of the current parent)
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

import pandas as pd
import numpy as np
from model.model import Model 
from data.load_data import load_dataset
from server.config import FEATURES
from dataset.adding_features import process_csvs_new


script_dir = Path(__file__).resolve().parent

def load_weights_from_npz(keras_model, npz_path):
    """
    Helper function to load raw NumPy arrays from an .npz file 
    into a Keras model instance.
    """
    with np.load(npz_path) as data:
        keys = sorted(data.files, key=lambda x: int(x.split('_')[1]))
        weights = [data[key] for key in keys]
    
    keras_model.set_weights(weights)

def run_inference(model_weights_path, input_csv_path, output_csv_path):
    """
    Used to generate predictions from a dataset not containing label
    """
    my_wrapper = Model(input_size=len(FEATURES))
    load_weights_from_npz(my_wrapper.model, model_weights_path)
    print(f"[System] Weights loaded from {model_weights_path} using .npz loader")

 
    # load data and compute eventual missing features
    # Data pre - processing is inside "load_dataset"
    X_train, X_test, X_val, y_train, y_test, y_val, full = load_dataset(
        input_csv_path,
        label_column="label",
        keep_features=FEATURES,
        drop_features=["hr_time_series", "resp_time_series", "stress_time_series"],
        test_size=0.2,
        val_size=0.1,
        load_label=0
    )

    X = full

    full_df = pd.DataFrame(X)
    full_df.to_csv("exported_full_data.csv", index=True, sep=';')

    my_wrapper.fit_scaler(X) 
    X_scaled = my_wrapper.scaler.transform(X).astype(np.float32)

    scaled_df = pd.DataFrame(X_scaled)
    scaled_df.to_csv("exported_full_scaled_data.csv", index=True, sep=';')

    # Sigmoid output
    y_raw = my_wrapper.model.predict(X_scaled, verbose=0)
    y_pred = (y_raw*100).flatten().astype(int)

    results_df = pd.DataFrame({
        'id': range(len(y_pred)),
        'label': y_pred
    })

    results_df.to_csv(output_csv_path, index=False)
    print(f"[Success] Predictions saved to {output_csv_path}")

run_inference(r"C:\Users\giaco\Documents\SCUOLA\Magistrale\25-26\BALANCE\checkpoints\global_model_round_35.npz",
              r"C:\Users\giaco\Documents\SCUOLA\Magistrale\25-26\BALANCE\test\x_test.csv", 
              r"predictions.csv")