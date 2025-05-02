# -*- coding: utf-8 -*-
# @Time    : 05/04/2025 14:39
# @Author  : mmai
# @FileName: DeepAR
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.Diffusion_Network4_UQ import NegativeBinomialDistributionLoss # Assuming this is the correct path and class
import numpy as np
import random
import time
import os
import pickle

# Define the DeepAR Model
class DeepAR(nn.Module):
    def __init__(self, input_size_per_node, hidden_size, num_nodes, num_layers, pred_horizon):
        """
        Simplified Multivariate DeepAR Model using LSTM.
        Assumes input feature at time t is the target value y_{t-1}.
        Predicts parameters for a Negative Binomial distribution for all nodes.
        """
        super().__init__()
        self.num_nodes = num_nodes
        # self.input_size_per_node = input_size_per_node # Typically 1 (target value)
        self.pred_horizon = pred_horizon # Number of future steps to predict (H)
        self.feature_size = input_size_per_node * num_nodes # LSTM input feature size

        # LSTM layer
        # Input shape: (N, L, H_in = feature_size)
        self.lstm = nn.LSTM(self.feature_size, hidden_size, num_layers, batch_first=True, dropout=0.5 if num_layers > 1 else 0)

        # Linear layers to output Negative Binomial parameters (mu and alpha) for all nodes
        self.fc_mu = nn.Linear(hidden_size, num_nodes)
        self.fc_alpha = nn.Linear(hidden_size, num_nodes)

        # Loss function (using the one from your UQ models)
        self.nll_loss_fn = NegativeBinomialDistributionLoss()

    def forward(self, x_hist, training=True, future_steps=None):
        """
        Forward pass for training or inference.

        Args:
            x_hist (Tensor): Historical target values [batch_size, history_length, num_nodes].
                             Used as input features y_0, ..., y_{t_0-1}.
            future_steps (int, optional): Number of future steps to predict autoregressively.
                                          Defaults to self.pred_horizon if None.

        Returns:
            mus (Tensor): Predicted means [batch_size, steps, num_nodes] (steps = history_length or future_steps)
            alphas (Tensor): Predicted alphas [batch_size, steps, num_nodes]
            lstm_hidden (tuple): Last hidden state (h_n, c_n) - useful for stateful prediction if needed.
        """
        batch_size = x_hist.size(0)
        history_length = x_hist.size(1) - 1
        steps_to_predict = future_steps if future_steps is not None else self.pred_horizon

        # Reshape input history for LSTM: [batch_size, history_length, num_nodes] -> [batch_size, history_length, feature_size]
        # get one step ahead of x_hist
        x_hist_ahead = x_hist[:, 1:, :]
        x_hist_lag = x_hist[:, :-1, :]
        lstm_input = x_hist_lag

        # Pass historical sequence through LSTM
        # lstm_out: [batch_size, history_length, hidden_size]
        # lstm_hidden: (h_n, c_n), each [num_layers, batch_size, hidden_size]
        lstm_out, lstm_hidden = self.lstm(lstm_input)

        # --- Training Mode (predict parameters for historical steps) ---
        if training:
            # Use output hidden states from history to predict parameters for y_1, ..., y_{t_0}
            mus_hist = []
            alphas_hist = []
            for t in range(history_length):
                h_t = lstm_out[:, t, :] # Hidden state at time t [batch_size, hidden_size]
                mu_t = F.softplus(self.fc_mu(h_t)) + 1e-6 # Ensure mu > 0 [batch_size, num_nodes]
                # mu_t = self.fc_mu(h_t)
                alpha_t = F.softplus(self.fc_alpha(h_t)) + 1e-6 # Ensure alpha > 0 [batch_size, num_nodes]
                mus_hist.append(mu_t)
                alphas_hist.append(alpha_t)

            mus = torch.stack(mus_hist, dim=1) # [batch_size, history_length, num_nodes]
            alphas = torch.stack(alphas_hist, dim=1) # [batch_size, history_length, num_nodes]
            return mus.clamp(min=0), alphas # Return params for history

        # --- Inference Mode (autoregressive prediction for future) ---
        else: # no sampling, just predict the mean and alpha
            mus_pred = []
            alphas_pred = []
            # Start prediction using the last known target value from history
            current_input = x_hist[:, -1, :].view(batch_size, 1, -1) # [batch_size, 1, feature_size]
            h_c_state = lstm_hidden # Use the final hidden state from history

            for _ in range(steps_to_predict):
                # Predict one step
                lstm_out_step, h_c_state = self.lstm(current_input, h_c_state)
                # lstm_out_step: [batch_size, 1, hidden_size]
                h_t = lstm_out_step.squeeze(1) # [batch_size, hidden_size]

                # Predict parameters for this step
                mu_t = F.softplus(self.fc_mu(h_t)) + 1e-6
                # mu_t = self.fc_mu(h_t)
                alpha_t = F.softplus(self.fc_alpha(h_t)) + 1e-6

                mus_pred.append(mu_t)
                alphas_pred.append(alpha_t)

                # Use the predicted mean as the input for the next step (autoregression)
                # Detach to prevent gradients flowing back through generated predictions
                current_input = mu_t.unsqueeze(1).detach() # Shape: [batch_size, 1, num_nodes] -> view for LSTM
                current_input = current_input.view(batch_size, 1, -1) # Shape: [batch_size, 1, feature_size]

            mus = torch.stack(mus_pred, dim=1) # [batch_size, pred_horizon, num_nodes]
            alphas = torch.stack(alphas_pred, dim=1) # [batch_size, pred_horizon, num_nodes]

            return mus.clamp(min=0), alphas # Return only future predictions during inference

    def compute_loss(self, mus, alphas, y_true):
        """
        Computes the Negative Binomial NLL loss.

        Args:
            mus (Tensor): Predicted means [batch_size, steps, num_nodes]
            alphas (Tensor): Predicted alphas [batch_size, steps, num_nodes]
            y_true (Tensor): Ground truth values [batch_size, steps, num_nodes]

        Returns:
            Tensor: Scalar loss value.
        """
        # Ensure inputs are valid for NLL loss
        mus = mus.clamp(min=1e-6)
        alphas = alphas.clamp(min=1e-6)
        y_true = y_true.clamp(min=0) # NB requires non-negative targets

        # Reshape for loss function if needed (assuming NLL averages over all elements)
        batch_size, steps, num_nodes = y_true.shape
        mus_flat = mus.reshape(-1, num_nodes)
        alphas_flat = alphas.reshape(-1, num_nodes)
        y_true_flat = y_true.reshape(-1, num_nodes)

        # The NLL loss class likely handles averaging
        loss = self.nll_loss_fn(mus_flat, y_true_flat, alphas_flat)
        return loss

    def inference(self, x_hist):
        """
        Performs inference (autoregressive prediction). Wrapper for forward in eval mode.

        Args:
            x_hist (Tensor): Historical target values [batch_size, history_length, num_nodes]

        Returns:
            mus (Tensor): Predicted means for the horizon [batch_size, pred_horizon, num_nodes]
            alphas (Tensor): Predicted alphas for the horizon [batch_size, pred_horizon, num_nodes]
        """
        self.eval() # Set model to evaluation mode
        with torch.no_grad():
            # Call forward in inference mode (y_target_hist=None)
            mus, alphas = self.forward(x_hist, training=False, future_steps=self.pred_horizon)
        return mus, alphas

# Main Offline Training Block (Adapted from LSTM_UQ.py)
if __name__ == '__main__':
    # Import necessary utilities (ensure paths are correct relative to this file)
    # Assuming lib and src are siblings of baselines or in PYTHONPATH
    try:
        from lib.utils import gen_data_dict, process_sensor_data, StandardScaler
        from lib.dataloader import FlowDataset
        from lib.utils import generating_ood_dataset, seperate_up_down, get_trainable_params_size
        from lib.metric import masked_rmse_np, masked_mae_np
        import dgl # Keep if generating_ood_dataset uses it, though DeepAR doesn't directly
        from dgl.data.utils import load_graphs
    except ImportError as e:
        print(f"Error importing utility functions: {e}")
        print("Please ensure 'lib' and 'src' directories are accessible.")
        exit(1)
    from torch.utils.data import DataLoader

    # --- Configuration ---
    random.seed(1)
    torch.manual_seed(1)
    np.random.seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset selection (similar to LSTM_UQ.py)
    parent_dir = '../sc_sensor' # Adjust if needed
    dataset_name = "train_station" # Example: "crossroad", "maze", "edinburgh"
    df_dict = {}

    # Data loading parameters based on dataset
    if dataset_name == "crossroad":
        train_sc = ['../sc_sensor/crossroad2']
        test_sc = ['../sc_sensor/crossroad1', '../sc_sensor/crossroad11', '../sc_sensor/crossroad13']
        lags = 5
        pred_horizon = 7
    elif dataset_name == "train_station":
        train_sc = ['../sc_sensor/train1']
        test_sc = ['../sc_sensor/train2']
        lags = 5
        pred_horizon = 7 # Actual number of steps to predict
    elif dataset_name == "maze":
        train_sc = ['../sc_sensor/maze19'] # Correct path if needed
        test_sc = ['../sc_sensor/maze13', '../sc_sensor/maze4']
        lags = 5
        pred_horizon = 7
        # Maze needs loading from pkl
    elif dataset_name == "edinburgh":
        train_sc = ['26Aug']
        test_sc = ['27Aug']
        lags = 6
        pred_horizon = 2 # Example, adjust as needed
        # Edinburgh needs loading from pkl

    print(f"Dataset: {dataset_name}, Lags: {lags}, Pred Horizon: {pred_horizon}")

    # Load and process data
    if dataset_name == "maze":
        pkl_path = "../sc_sensor/maze/flow_data.pkl" # Adjust path if necessary
        print(f"Loading data from {pkl_path}")
        if not os.path.exists(pkl_path): raise FileNotFoundError(f"Cannot find {pkl_path}")
        with open(pkl_path, "rb") as f:
            data_dict = pickle.load(f)
    elif dataset_name == "edinburgh":
         pkl_path = "../sc_sensor/edinburgh/flow_data_edinburgh.pkl" # Adjust path if necessary
         print(f"Loading data from {pkl_path}")
         if not os.path.exists(pkl_path): raise FileNotFoundError(f"Cannot find {pkl_path}")
         with open(pkl_path, "rb") as f:
             data_dict = pickle.load(f)
    else:
        print(f"Processing sensor data from {parent_dir}...")
        df_dict = process_sensor_data(parent_dir, df_dict)
        data_dict = gen_data_dict(df_dict)
        data_dict = seperate_up_down(data_dict) # Keep if necessary for data generation

    # Generate datasets (OOD or In-sample)
    # IMPORTANT: generating_ood_dataset needs to return x_*, y_* with shapes:
    # x: [num_samples, lags, num_nodes] (History)
    # y: [num_samples, pred_horizon, num_nodes] (Future Targets)
    print("Generating train/val/test splits...")
    x_train, y_train, x_val, y_val, x_test, y_test = generating_ood_dataset(
        data_dict, train_sc, test_sc, lags=lags, horizons=pred_horizon, shuffle=True
    )
    # x_train, y_train, x_val, y_val, x_test, y_test = generating_insample_dataset(data_dict, train_sc,
    #                                                                              lags=lags,
    #                                                                              horizons=pred_horizon,
    #                                                                              portion=0.7,
    #                                                                              shuffle=True)

    print(f"Data shapes: x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"             x_val: {x_val.shape}, y_val: {y_val.shape}")
    print(f"             x_test: {x_test.shape}, y_test: {y_test.shape}")

    # Get number of nodes and verify prediction horizon matches y_train
    num_nodes = x_train.shape[2]
    # actual_pred_horizon = y_train.shape[1]
    # if actual_pred_horizon != pred_horizon:
    #      print(f"Warning: Mismatch between configured pred_horizon ({pred_horizon}) and generated data ({actual_pred_horizon}). Using {actual_pred_horizon}.")
    #      pred_horizon = actual_pred_horizon # Use actual horizon from data

    # Create DataLoaders
    # batch_size = 32 # Example
    num_input_timesteps = x_train.shape[1] # number of input time steps
    num_nodes = x_train.shape[2] # number of ancestor nodes, minus the down stream node

    train_dataset = FlowDataset(x_train,
                                y_train, batch_size=16)
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    val_dataset = FlowDataset(x_val,
                             y_val, batch_size=16)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    test_dataset = FlowDataset(x_test,
                              y_test, batch_size=16)
    test_dataloader = DataLoader(test_dataset, batch_size=16)
    # set seed
    torch.manual_seed(1)

    # --- Model Initialization ---
    model = DeepAR(input_size_per_node=1, # Assuming input feature is just the target value
                   hidden_size=64,        # Example hyperparameter
                   num_nodes=num_nodes,
                   num_layers=2,          # Example hyperparameter
                   pred_horizon=pred_horizon-1)
    model.to(device)
    print(f"DeepAR Model Initialized. Trainable parameters: {get_trainable_params_size(model)}")

    # --- Training Setup ---
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    num_epochs = 1000 # Example

    print("Starting training...")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10 # Early stopping patience

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        start_time_epoch = time.time()

        for i, (x_batch, y_batch) in enumerate(train_dataloader):
            # x_batch: [batch_size, lags, num_nodes] (History y_0..y_{t0-1})
            # y_batch: [batch_size, pred_horizon, num_nodes] (Future targets y_t0..y_{t0+H-1})
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            if i == 16:
                pass

            # --- Training Loss Calculation ---
            # Predict future steps based on the history x_batch
            # Call forward in inference mode (y_target_hist=None) to get future predictions
            # Ensure it predicts the correct number of steps (pred_horizon)
            # The inference mode returns only mus and alphas for the future.
            mus_pred, alphas_pred = model(torch.cat((x_batch, y_batch), dim=1), training=True, future_steps=None)

            # Compute loss on the predicted future steps vs the actual future targets
            # Now mus_pred/alphas_pred and y_batch have the same shape [batch_size, pred_horizon, num_nodes]
            loss = model.compute_loss(mus_pred, alphas_pred, torch.cat((x_batch, y_batch), dim=1)[:, 1:, :])

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss encountered at epoch {epoch}, batch {i}. Skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        epoch_time = time.time() - start_time_epoch

        # --- Validation Step ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_val_batch, y_val_batch in val_dataloader:
                x_val_batch = x_val_batch.to(device)
                y_val_batch = y_val_batch.to(device)
                mus_val, alphas_val = model.inference(x_val_batch)
                val_loss = model.compute_loss(mus_val, alphas_val, y_val_batch)
                if not (torch.isnan(val_loss) or torch.isinf(val_loss)):
                    total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_dataloader)

        print(f'Epoch: {epoch+1}/{num_epochs}, Time: {epoch_time:.2f}s, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # --- Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the best model
            save_dir = '../checkpoint/deepar'
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'deepar_{dataset_name}_lags{lags}_hor{pred_horizon}_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Validation loss improved. Saved model to {save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Validation loss did not improve for {patience} epochs. Early stopping.")
                break

    print("Training finished.")

    # --- Testing Step ---
    print("Loading best model for testing...")
    # Load the best model saved during training
    best_model_path = os.path.join('../checkpoint/deepar', f'deepar_{dataset_name}_lags{lags}_hor{pred_horizon}_best.pth')
    if os.path.exists(best_model_path):
         model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
         print("Warning: Best model checkpoint not found. Testing with the last state.")

    model.eval()
    test_loss_nll = 0
    test_loss_mse = 0 # MSE on mean prediction
    all_preds_mu = []
    all_labels = []

    print("Starting testing...")
    with torch.no_grad():
        for x_test_batch, y_test_batch in test_dataloader:
            x_test_batch = x_test_batch.to(device)
            y_test_batch = y_test_batch.to(device)

            mus_test, alphas_test = model.inference(x_test_batch) # [batch, horizon, nodes]

            # Calculate NLL loss
            nll = model.compute_loss(mus_test, alphas_test, y_test_batch)
            if not (torch.isnan(nll) or torch.isinf(nll)):
                test_loss_nll += nll.item()

            # Calculate MSE loss on the mean prediction
            mse = F.mse_loss(mus_test, y_test_batch)
            if not (torch.isnan(mse) or torch.isinf(mse)):
                test_loss_mse += mse.item()

            all_preds_mu.append(mus_test.cpu().numpy())
            all_labels.append(y_test_batch.cpu().numpy())

    avg_test_nll = test_loss_nll / len(test_dataloader)
    avg_test_mse = test_loss_mse / len(test_dataloader)

    print(f'Test NLL: {avg_test_nll:.4f}')
    print(f'Test MSE (on mean): {avg_test_mse:.4f}')

    # Concatenate results for metric calculation
    all_preds_mu = np.concatenate(all_preds_mu, axis=0) # [total_samples, horizon, nodes]
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate other metrics (RMSE, MAE) - requires selecting relevant nodes (e.g., dst_idx)
    # Need graph info for dst_idx if calculating metrics only on specific nodes
    try:
        g_data = load_graphs('../graphs/4graphs.bin') # Adjust path if needed
        if dataset_name == "crossroad": g = g_data[0][0]
        elif dataset_name == "train_station": g = g_data[0][1]
        elif dataset_name == "maze": g = g_data[0][2]
        elif dataset_name == "edinburgh": g = g_data[0][3]
        else: g = None

        if g:
            src, dst = g.edges()
            dst_idx = dst.unique().cpu().numpy()
            print(f"Calculating metrics for destination nodes: {dst_idx}")
            # Calculate metrics only for destination nodes
            rmse = masked_rmse_np(all_preds_mu[:, :, dst_idx], all_labels[:, :, dst_idx])
            mae = masked_mae_np(all_preds_mu[:, :, dst_idx], all_labels[:, :, dst_idx])
            print(f'Test RMSE (Dst Nodes): {rmse:.4f}')
            print(f'Test MAE (Dst Nodes): {mae:.4f}')

            # Calculate metrics for each step ahead
            for step in range(pred_horizon):
                 rmse_step = masked_rmse_np(all_preds_mu[:, step, dst_idx], all_labels[:, step, dst_idx])
                 mae_step = masked_mae_np(all_preds_mu[:, step, dst_idx], all_labels[:, step, dst_idx])
                 print(f'  Step {step+1} - RMSE: {rmse_step:.4f}, MAE: {mae_step:.4f}')

        else:
             print("Graph not loaded for dataset. Cannot calculate metrics for specific nodes.")
             # Calculate metrics for all nodes
             rmse_all = masked_rmse_np(all_preds_mu, all_labels)
             mae_all = masked_mae_np(all_preds_mu, all_labels)
             print(f'Test RMSE (All Nodes): {rmse_all:.4f}')
             print(f'Test MAE (All Nodes): {mae_all:.4f}')

    except Exception as e:
        print(f"Could not load graph or calculate node-specific metrics: {e}")

    print("Offline DeepAR script finished.")

