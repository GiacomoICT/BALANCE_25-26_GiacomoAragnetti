import os
import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg, FedProx
from typing import Dict
from .config import (
    MIN_CLIENTS, MIN_FIT_CLIENTS, MIN_EVALUATE_CLIENTS,
    FRACTION_FIT, FRACTION_EVALUATE,
    LOCAL_EPOCHS, BATCH_SIZE, MODEL_DIR
)

os.makedirs(MODEL_DIR, exist_ok=True)

class SaveModelStrategy(FedProx):

    def on_fit_config_fn(self, server_round: int) -> Dict[str, int]:
        """Sends training parameters to clients."""
        return {"local_epochs": LOCAL_EPOCHS, "batch_size": BATCH_SIZE, "proximal_mu":self.proximal_mu}

    def aggregate_fit(self, server_round, results, failures):
        """Aggregates weights and saves the global model to disk."""
        aggregated_parameters, metrics_aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
            # Save by round number so you can track progress or revert if needed
            file_path = os.path.join(MODEL_DIR, f"global_model_round_{server_round}.npz")
            np.savez(file_path, *weights)
            print(f"[Server] Saved global model for round {server_round} to {file_path}")

        return aggregated_parameters, metrics_aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregates regression metrics (MAE and Accuracy Threshold)."""
        # Call base to get the aggregated loss (which is Huber in our Model)
        aggregated_loss, _ = super().aggregate_evaluate(server_round, results, failures)

        if not results:
            return aggregated_loss, {}

        total_examples = 0
        weighted_mae_sum = 0.0
        weighted_acc_sum = 0.0

        for _, eval_res in results:
            n = eval_res.num_examples
            total_examples += n

            # 1. Aggregate MAE (Mean Absolute Error)
            mae = eval_res.metrics.get("mae")
            if mae is not None:
                weighted_mae_sum += mae * n

            # 2. Aggregate Accuracy (Within +/- 5 points)
            acc = eval_res.metrics.get("accuracy")
            if acc is not None:
                weighted_acc_sum += acc * n

        # Compute weighted averages
        final_mae = weighted_mae_sum / total_examples if total_examples > 0 else 0.0
        final_acc = weighted_acc_sum / total_examples if total_examples > 0 else 0.0

        print(f"[Server] Round {server_round} Evaluation:")
        print(f"  >> Global Huber (Loss): {aggregated_loss:.4f}")
        print(f"  >> Global MAE: {final_mae:.4f} points")
        print(f"  >> Accuracy (+/- 5 pts): {final_acc * 100:.2f}%")

        return aggregated_loss, {"mae": final_mae, "accuracy": final_acc}

def get_strategy() -> SaveModelStrategy:
    return SaveModelStrategy(
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=FRACTION_EVALUATE,
        min_available_clients=MIN_CLIENTS,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_EVALUATE_CLIENTS,

        proximal_mu=1
    )