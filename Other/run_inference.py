import pandas as pd
import numpy as np
from model.model import Model  # Ensure this path matches your file structure
from data.load_data import load_dataset
from server.config import FEATURES
from dataset.adding_features import process_csvs_new

def load_weights_from_npz(keras_model, npz_path):
    """
    Helper function to load raw NumPy arrays from an .npz file 
    into a Keras model instance.
    """
    with np.load(npz_path) as data:
        # Flower's np.savez saves weights as arr_0, arr_1, etc.
        # We sort them by index to ensure they match the model's layer order
        keys = sorted(data.files, key=lambda x: int(x.split('_')[1]))
        weights = [data[key] for key in keys]
    
    # Inject the arrays directly into the model
    keras_model.set_weights(weights)

def run_inference(model_weights_path, input_csv_path, output_csv_path):
    """
    Uses the custom Model class to generate predictions from a CSV 
    using weights saved in .npz format.
    """
    # 1. Initialize the custom Model wrapper
    # This sets up the architecture (12-6 nodes) and internal scaler

    my_wrapper = Model(input_size=len(FEATURES))


    
    # 2. UPDATED: Load the trained weights from .npz
    # We access the internal .model (Keras object) and use our helper
    load_weights_from_npz(my_wrapper.model, model_weights_path)
    print(f"[System] Weights loaded from {model_weights_path} using .npz loader")

    # 3. Load input data
    # Assuming CSV has the same 6 features used during training
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


    # 4. Handle Scaling
    # IMPORTANT: Using fit_scaler on new data is a fallback. 
    # For perfect accuracy, the scaler should ideally be loaded from training
    my_wrapper.fit_scaler(X) 
    X_scaled = my_wrapper.scaler.transform(X).astype(np.float32)

    scaled_df = pd.DataFrame(X_scaled)
    scaled_df.to_csv("exported_full_scaled_data.csv", index=True, sep=';')

    # 5. Predict (Raw Sigmoid 0-1)
    # The sigmoid output ensures values are between 0 and 1
    y_raw = my_wrapper.model.predict(X_scaled, verbose=0)

    # 6. Post-process: Convert 0-1 back to 1-100 scale
    # Matches the scaling logic used in your evaluate method
    y_pred = (y_raw*100).flatten().astype(int)

    # 7. Format and Save as [id, label]
    results_df = pd.DataFrame({
        'id': range(len(y_pred)),
        'label': y_pred
    })

    results_df.to_csv(output_csv_path, index=False)
    print(f"[Success] Predictions saved to {output_csv_path}")

# Example Usage
print("LENGTH OF FEATURES VECTOR")
print(len(FEATURES))
run_inference("checkpoints\global_model_round_35.npz", "x_test.csv", "predictions.csv")