import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataset.adding_features import process_df

def load_dataset(
    csv_path: str, 
    label_column: str = "label", 
    keep_features: list = None, 
    drop_features: list = None,
    test_size: float = 0.2, 
    val_size: float = 0.1,
    load_label = 1,
):
    # 1. Load the data
    try:
        df = pd.read_csv(csv_path, sep=';', on_bad_lines='skip', engine='c', low_memory=False)
    except Exception as e:
        print(f"[error] Failed to read CSV at {csv_path}: {e}")
        raise

    df = process_df(df)

    # 2. Separate Label (y)
    if load_label == 1:
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found")
        y = df[label_column].values
        X_df = df.drop(columns=[label_column])
    else:
        y = df.values[:,1] if df.shape[1] > 1 else np.zeros(len(df))
        X_df = df

    # 3. Feature Selection
    if keep_features:
        valid_keeps = [f for f in keep_features if f in X_df.columns]
        X_df = X_df[valid_keeps]
    if drop_features:
        X_df = X_df.drop(columns=[f for f in drop_features if f in X_df.columns])

    # --- STEP 3.5: MEAN IMPUTATION ---
    # Ensure all columns are numeric (convert strings to NaN if necessary)
    X_df = X_df.apply(pd.to_numeric, errors='coerce')
    
    # Fill missing values (NaN) with the mean of their respective columns
    X_df = X_df.fillna(X_df.mean())
    
    # Optional: If a whole column is NaN, fill with 0 to prevent downstream errors
    X_df = X_df.fillna(0)
    # --------------------------------

    # 4. Convert to values and Split
    X = X_df.values
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=50
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=50
    )
    
    return X_train, X_test, X_val, y_train, y_test, y_val, X