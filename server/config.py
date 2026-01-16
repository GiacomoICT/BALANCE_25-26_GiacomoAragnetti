"""Server configuration constants.

Keep training / federation-related constants here. These variables are
imported by `server/server_flwr.py` and `server/strategy.py`.
"""

# Number of federated learning rounds to run
NUM_ROUNDS = 35

# Address the Flower server listens on (host:port)
SERVER_ADDRESS = "localhost:8080"

# Directory where the server persists the global model
MODEL_DIR = "checkpoints"

# Minimum available clients and fit/evaluate minimums
MIN_CLIENTS = 8
MIN_FIT_CLIENTS = 8
MIN_EVALUATE_CLIENTS = 8

# Local training defaults sent to clients
LOCAL_EPOCHS = 8
BATCH_SIZE = 16

# Fraction of clients used for fit/evaluate each round (1.0 = all available)
FRACTION_FIT = 1.0
FRACTION_EVALUATE = 1.0


FEATURES = ["ts_hr_circadian_delta", "ts_hr_recovery_slope", "resp_avgTomorrowSleepRespirationValue", "act_totalCalories","ts_hr_stress_coupling","ts_stress_load","ratio_deep_sleep","ts_resp_instability"]


"""
Best results: 

So far, the best results have been obtained with 
12 - 6 - 1 MLP with low dropouts and sigmoid activation function 


Low dropouts outperforming seem to indicate that increasing the capacity of the 
network would be appropriate, as well as to validation analysis

Very small batches are outperformed by 16/20. Local epochs around 5 
Plateau is found after around 30 rounds.

High client variance in results seem to indicate that fedAvg could actually be 
substituted by smarter weighting --> I want to look at label distribution as well

Maybe some nodes have really poor correlation. Either move to Huber Loss (MSE for small,
MAE for high) which means that we are not sending MSE weights updates

"""