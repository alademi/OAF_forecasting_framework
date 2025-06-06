import copy
import os
import numpy as np
import torch
import joblib
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader

import util
from models_config import ModelBuilder
from underopt_models_config import UnderOptModelBuilder

HORIZON = 1
WINDOW_SIZE = 7
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.0001
PATIENCE = 5
SEED = 42


def clone_model(model):
    cloned_model = copy.deepcopy(model)
    return cloned_model


def train_model(train, test, model_name, file_name, WINDOW_SIZE=7, HORIZON=1):
    train_windows, train_labels = util.make_windows(train, WINDOW_SIZE, HORIZON)
    test_windows, test_labels = util.make_windows(test, WINDOW_SIZE, HORIZON)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = f'Init_Models/{file_name}/base-model/{model_name}'
    os.makedirs(model_path, exist_ok=True)

    if model_name in ["mlp", "decision_tree", "random_forest", "xgboost"]:
        if train_windows.ndim == 3 and train_windows.shape[2] == 1:
            train_windows_input = np.squeeze(train_windows, axis=2)
            test_windows_input = np.squeeze(test_windows, axis=2)
        else:
            train_windows_input = train_windows
            test_windows_input = test_windows
    else:
        if train_windows.ndim == 2:
            train_windows_input = np.expand_dims(train_windows, axis=-1)
            test_windows_input = np.expand_dims(test_windows, axis=-1)
        else:
            train_windows_input = train_windows
            test_windows_input = test_windows

    if model_name in ["decision_tree", "random_forest", "xgboost"]:
        model_builder = ModelBuilder(model_type=model_name, n_timesteps=WINDOW_SIZE, horizon=HORIZON)
        model = model_builder.build_model()
        model.fit(train_windows_input, train_labels)
        checkpoint_path = os.path.join(model_path, "best_model.pkl")
        joblib.dump(model, checkpoint_path)
        print(f"Saved {model_name} model to: {checkpoint_path}")
        predictions = model.predict(test_windows_input)
        predictions = predictions.reshape(-1, 1)
        if predictions.ndim != 2 or predictions.shape[1] != 1:
            raise ValueError(f"Error in the prediction shape: {predictions.shape}")
        else:
            print(f"Prediction shape OK: {predictions.shape}")
        return

    model_builder = ModelBuilder(model_type=model_name, n_timesteps=WINDOW_SIZE, horizon=HORIZON)
    model = model_builder.build_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()

    train_tensor_x = torch.tensor(train_windows_input, dtype=torch.float32)
    train_tensor_y = torch.tensor(train_labels, dtype=torch.float32)
    test_tensor_x = torch.tensor(test_windows_input, dtype=torch.float32)
    test_tensor_y = torch.tensor(test_labels, dtype=torch.float32)

    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    test_dataset = TensorDataset(test_tensor_x, test_tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_val_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(test_dataset)
        print(
            f"Model: {model_name} | Epoch [{epoch}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(model_path, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved best model to: {checkpoint_path}")

    return model


def load_model(model_name, path):
    """
    Loads a trained model from the given path.

    Args:
        model_name (str): Name of the model to be loaded.
        path (str): Path to the directory where the model is stored.

    Returns:
        model: Loaded model.
    """
    model_path = os.path.join(path, "best_model.pkl" if model_name in ["decision_tree", "random_forest",
                                                                       "xgboost"] else "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Load scikit-learn models
    if model_name in ["decision_tree", "random_forest", "xgboost"]:
        model = joblib.load(model_path)
        print(f"Loaded {model_name} model from: {model_path}")
        return model

    # Load PyTorch models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_builder = ModelBuilder(model_type=model_name, n_timesteps=WINDOW_SIZE, horizon=HORIZON)
    model = model_builder.build_model().to(device)

    # Load state dict safely with weights_only=True
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    model.eval()
    print(f"Loaded {model_name} model from: {model_path}")

    return model


def load_underopt_model(model_name, path):
    """
    Loads a trained model from the given path.

    Args:
        model_name (str): Name of the model to be loaded.
        path (str): Path to the directory where the model is stored.

    Returns:
        model: Loaded model.
    """
    model_path = os.path.join(path, "best_model.pkl" if model_name in ["decision_tree", "random_forest",
                                                                       "xgboost"] else "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Load scikit-learn models
    if model_name in ["decision_tree", "random_forest", "xgboost"]:
        model = joblib.load(model_path)
        print(f"Loaded {model_name} model from: {model_path}")
        return model

    # Load PyTorch models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_builder = UnderOptModelBuilder(model_type=model_name, n_timesteps=WINDOW_SIZE, horizon=HORIZON)
    model = model_builder.build_model().to(device)

    # Load state dict safely with weights_only=True
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    model.eval()
    print(f"Loaded {model_name} model from: {model_path}")

    return model



def evaluate(model, model_name, subsequence):
    """
    Evaluate the model on a single subsequence and return the prediction.
    """
    subseq = np.array(subsequence)

    if hasattr(model, 'predict'):
        # Classical model (e.g., Scikit-learn models)
        if subseq.ndim == 1:
            subseq = subseq.reshape(1, -1)
        elif subseq.ndim == 2 and subseq.shape[1] == 1:
            subseq = np.squeeze(subseq, axis=-1).reshape(1, -1)
        pred = model.predict(subseq)
        pred = np.squeeze(pred)

    else:
        # PyTorch-based model
        device = next(model.parameters()).device

        if model_name == "mlp":
            # MLP expects 2D input: (batch, WINDOW_SIZE)
            if subseq.ndim == 1:
                subseq = np.expand_dims(subseq, axis=0)  # Ensure batch dimension
            input_tensor = torch.tensor(subseq, dtype=torch.float32, device=device)

        else:
            # CNN-based models expect 3D input: (batch, sequence_length, channels)
            if subseq.ndim == 1:
                subseq = np.expand_dims(subseq, axis=0)  # Add batch dim


            # 🔹 **Ensure Shape is (batch_size, sequence_length, channels)**
            subseq = subseq.reshape(1, 7, 1)

            # **Do NOT swap axes this time** — because the issue was that channels and sequence_length were being mixed up.
            # subseq = np.transpose(subseq, (0, 1, 2))  # No need for this swap anymore

            # Convert to PyTorch tensor
            input_tensor = torch.tensor(subseq, dtype=torch.float32, device=device)


        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]  # Extract first element if model returns a tuple
        pred = output.cpu().numpy().squeeze()

    # If the model is "mq-cnn", select the median quantile (assumed at index 1).
    if model_name == "mq-cnn":
        pred = np.atleast_1d(pred)
        if pred.shape[0] == 3:
            pred = pred[1]

    return pred




def load_finetuned_models(model_class, model_path, device):
    """
    Load the model and apply proper settings.
    """
    model = model_class()
    model.load_state_dict(torch.load(model_path, weights_only=True))  # Explicitly load weights only
    model.to(device)
    model.eval()
    return model


def train_specialized_models(base_model, cluster_windows, cluster_labels, save_dir):
    if len(cluster_windows) < 2 or len(cluster_labels) < 2:
        print(f"Skipping training for cluster in {save_dir} due to insufficient data.")
        return None

    if isinstance(base_model, torch.nn.Module):
        cloned_model = copy.deepcopy(base_model)
        cloned_model.load_state_dict(base_model.state_dict())
    else:
        cloned_model = clone_model(base_model)

    if isinstance(cloned_model, BaseEstimator):
        if cluster_windows.ndim == 3 and cluster_windows.shape[2] == 1:
            cluster_windows = np.squeeze(cluster_windows, axis=2)
        X_train, X_val, y_train, y_val = train_test_split(cluster_windows, cluster_labels, test_size=0.2, random_state=SEED)
        cloned_model.fit(X_train, y_train)
        val_loss = np.mean((cloned_model.predict(X_val) - y_val) ** 2)
        print(f"Validation Loss: {val_loss}")
        os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(save_dir, "best_model.pkl")
        joblib.dump(cloned_model, model_save_path)
        return cloned_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cloned_model.to(device)

    if cluster_windows.ndim == 2:
        cluster_windows = np.expand_dims(cluster_windows, axis=-1)

    cluster_windows = torch.tensor(cluster_windows, dtype=torch.float32).to(device)
    cluster_labels = torch.tensor(cluster_labels, dtype=torch.float32).to(device)

    if len(cluster_windows) < 2:
        print(f"Skipping training for cluster in {save_dir} due to insufficient data after tensor conversion.")
        return None

    # Ensure at least 2 samples are available
    train_size = int(0.8 * len(cluster_windows))
    if train_size < 1 or (len(cluster_windows) - train_size) < 1:
        print(f"Skipping training for cluster in {save_dir} because train or validation set would be empty.")
        return None

    train_data, val_data = torch.utils.data.random_split(
        TensorDataset(cluster_windows, cluster_labels), [train_size, len(cluster_windows) - train_size]
    )

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

    optimizer = optim.Adam(cloned_model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float("inf")
    patience_counter = 0

    cloned_model.train()
    for epoch in range(EPOCHS):
        cloned_model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = cloned_model(batch_x)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if outputs.shape != batch_y.shape:
                batch_y = batch_y.view_as(outputs)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if len(val_loader) == 0:
            print(f"Skipping validation epoch in {save_dir} because val_loader is empty.")
            return None
        val_loss = 0.0
        cloned_model.eval()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = cloned_model(batch_x)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                val_loss += criterion(outputs, batch_y).item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {epoch_loss / len(train_loader)}, Val Loss: {val_loss}")

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "best_model.pth")
    torch.save(cloned_model.state_dict(), model_save_path)
    print(f"Fine-tuned model saved at: {model_save_path}")
    return cloned_model







