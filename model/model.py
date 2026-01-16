import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Sequence

def _sanitize_X(X):
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

def _find_leaky_columns(X_tr: np.ndarray, y_tr: np.ndarray, threshold: float = 0.999) -> Sequence[int]:
    if threshold is None or threshold <= 0:
        return []
    df = pd.DataFrame(X_tr)
    yv = np.asarray(y_tr).ravel()
    to_drop = []
    for col in df.columns:
        v = df[col].values
        if np.array_equal(v, yv):
            to_drop.append(col)
            continue
        if np.std(v) > 0:
            corr = np.corrcoef(v, yv)[0, 1]
            if np.isfinite(corr) and abs(corr) >= threshold:
                to_drop.append(col)
    return to_drop

class WeightChangeLogger(Callback):
    def on_train_begin(self, logs=None):
        self.prev_weights = [w.numpy().copy() for w in self.model.weights]

    def on_epoch_end(self, epoch, logs=None):
        deltas = []
        for w, prev in zip(self.model.weights, self.prev_weights):
            arr = w.numpy()
            deltas.append(np.linalg.norm(arr - prev))
        self.prev_weights = [w.numpy().copy() for w in self.model.weights]
        tot_delta = float(np.sum(deltas))
        print(f"[diag] L2 weight delta epoch {epoch+1}: {tot_delta:.8f}")

class RegressionMetricsLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_mae = logs.get('mae')
        val_mae = logs.get('val_mae')
        t_mae = train_mae if train_mae is not None else 0.0
        v_mae = val_mae if val_mae is not None else 0.0
        print(f"Epoch {epoch + 1}: Train MAE = {t_mae:.4f}, Val MAE = {v_mae:.4f}")

class Model:
    def __init__(self, input_size):
        self.input_size = input_size
        self.scaler = RobustScaler()
        self._scaler_fitted = False

        # --- SIGMOID ARCHITECTURE ---
        self.model = Sequential([
            Dense(12, activation='leaky_relu', input_shape=(input_size,), 
                  kernel_regularizer=regularizers.l2(0.001)), 
            Dropout(0.4), 
            
            Dense(6, activation='leaky_relu', 
                  kernel_regularizer=regularizers.l2(0.001)),
            Dropout(0.2),
            
            # Back to Sigmoid (Outputs 0.0 to 1.0)
            Dense(1, activation='sigmoid') 
        ])  

        self.model.compile(
            optimizer=Adam(learning_rate=0.001, clipnorm=1.0), 
            loss='mean_squared_error', 
            metrics=['mae'],
            run_eagerly=False,
        )

    def fit_scaler(self, X):
        X = _sanitize_X(X)
        self.scaler.fit(X)
        self._scaler_fitted = True

    def _ensure_scaler(self, X):
        if not self._scaler_fitted:
            self.fit_scaler(X)

    def fit(self, X_tr, y_tr, X_va, y_va, epochs=20, batch_size=32, leak_corr_threshold=0.999):
        # 1. Sanitize and Scale Labels (CRITICAL for Sigmoid)
        X_tr = _sanitize_X(X_tr)
        X_va = _sanitize_X(X_va)
        y_tr = np.asarray(y_tr, dtype=np.float32).reshape(-1, 1) / 100.0
        y_va = np.asarray(y_va, dtype=np.float32).reshape(-1, 1) / 100.0

        if not self._scaler_fitted:
            X_combined = np.vstack([X_tr, X_va])
            self.fit_scaler(X_combined)

        # 2. Leak Guard
        drop_cols = _find_leaky_columns(X_tr, y_tr, threshold=leak_corr_threshold)
        if drop_cols:
            keep_mask = np.ones(X_tr.shape[1], dtype=bool)
            keep_mask[np.array(drop_cols, dtype=int)] = False
            X_tr, X_va = X_tr[:, keep_mask], X_va[:, keep_mask]

        # 3. Feature Scaling
        X_tr_scaled = self.scaler.transform(X_tr).astype(np.float32)
        X_va_scaled = self.scaler.transform(X_va).astype(np.float32)

        # 4. Training Callbacks
        early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        #reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=0.0001)
        
        return self.model.fit(
            X_tr_scaled, y_tr,
            validation_data=(X_va_scaled, y_va),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=[early_stop, WeightChangeLogger(), RegressionMetricsLogger()],
            verbose=1
        )

    def evaluate(self, X, y):
        X = _sanitize_X(X)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1) # Keep real scale for comparison
        
        self._ensure_scaler(X)
        X_scaled = self.scaler.transform(X).astype(np.float32)

        # Predict (0.0 - 1.0) and convert back to (1 - 100)
        y_raw = self.model.predict(X_scaled, verbose=0)
        y_pred = y_raw * 100.0

        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        within_threshold = np.abs(y.flatten() - y_pred.flatten()) < 5.0
        accuracy_threshold = np.mean(within_threshold)

        return float(mse), float(accuracy_threshold), float(mae)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)