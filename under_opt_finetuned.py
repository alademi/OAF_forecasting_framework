import copy
import csv
import math
import random
import os
import time

import joblib
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import StandardScaler
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader

import models_util
import util
from underopt_models_config import UnderOptModelBuilder
from models_util import load_model, load_underopt_model
from util import perform_clustering

HORIZON = 1
WINDOW_SIZE = 7
BATCH_SIZE = 128
EPOCHS = 50
LR = 0.001
SEED = 42

MODELS = UnderOptModelBuilder.get_available_models()
scaler = StandardScaler()

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reset_pytorch_configuration():
    """
    Reset PyTorch configuration by clearing CUDA cache and re-setting random seeds.
    This helps avoid interference from previous runs when processing multiple datasets.
    """
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Reinitialize random seeds for reproducibility
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def preprocess_data(data):
    # First split: 80% Train+Validation, 20% Test
    first_split = int(0.8 * len(data))
    train_val, test = data[:first_split], data[first_split:]

    # Second split: 50% Training, 50% Validation from Train+Validation
    second_split = int(0.5 * len(train_val))
    train, val = train_val[:second_split], train_val[second_split:]

    norm_values = train_val.iloc[:,
                  1].values  # train + validation data to normalize the data without the test data to avoid data leakage
    scaler.fit(norm_values.reshape(-1, 1))  # fit the scaler with the values of the training and validation

    train_values = train.iloc[:, 1].values
    train_norm = scaler.transform(train_values.reshape(-1, 1))

    val_values = val.iloc[:, 1].values
    val_norm = scaler.transform(val_values.reshape(-1, 1))

    test_values = test.iloc[:, 1].values
    test_norm = scaler.transform(test_values.reshape(-1, 1))

    return train_norm, val_norm, test_norm


def cluster_data(val):
    val_windows, val_labels = util.make_windows(val, WINDOW_SIZE, HORIZON)
    clustering_results = perform_clustering(val_windows)
    return clustering_results


def finetune_models(base_model, model_name, dataset, val, clustering_results):
    clusters_no = clustering_results.n_clusters
    clusters_labels = clustering_results.labels_

    val_windows, val_labels = util.make_windows(val, WINDOW_SIZE, HORIZON)

    clustered_windows, clustered_labels = util.create_clustered_data(val_windows, val_labels, clusters_no,
                                                                     clusters_labels)

    finetuned_models = []
    for i in range(clusters_no):
        save_dir = f"/Users/aalademi/PycharmProjects/ecml/Models/{dataset}/offline/{model_name}/cluster{i + 1}"
        cluster_windows = clustered_windows[i]
        cluster_labels = clustered_labels[i]
        cluster_model = models_util.train_specialized_models(base_model, cluster_windows, cluster_labels, save_dir)
        finetuned_models.append(cluster_model)

    return finetuned_models


def get_base_model(model_name, dataset_name):
    path = f"/Users/aalademi/PycharmProjects/ecml/Models/{dataset_name}/base-model/{model_name}"
    model = load_underopt_model(model_name, path)
    return model


def evaluate_model(base_model, model_name, test, dataset_name, original_data):
    test_windows, test_labels = util.make_windows(test, WINDOW_SIZE, HORIZON)
    predictions = []

    for window in test_windows:
        prediction = models_util.evaluate(base_model, model_name, window)
        predictions.append(prediction)

    predictions = np.array(predictions)
    labels = np.array(test_labels).squeeze()

    # Inverse-transform to the original scale.
    predictions_orig = scaler.inverse_transform(predictions.reshape(-1, 1)).squeeze()
    labels_orig = scaler.inverse_transform(labels.reshape(-1, 1)).squeeze()
    print("Predictions Shape:", predictions_orig.shape)
    print("Labels Shape:", labels_orig.shape)

    mse_orig = np.mean((predictions_orig - labels_orig) ** 2)
    rmse_orig = np.sqrt(mse_orig)
    nrmse_orig = normalize_rmse(rmse_orig,original_data)

    print("##########################")
    print(f"Model: {model_name}")
    print(f"Predictions (original scale): {predictions_orig}")
    print("##########################")

    print(f"mse: {mse_orig}")

    return rmse_orig, nrmse_orig


def get_finetuned_models(model_name, dataset_name, clusters_no):
    finetuned_models = []
    for i in range(clusters_no):
        path = f"/Users/aalademi/PycharmProjects/ecml/Models/{dataset_name}/offline/{model_name}/cluster{i + 1}"
        model = load_underopt_model(model_name, path)
        finetuned_models.append(model)

    return finetuned_models



def retrigger_clustering(kmeans, validation_set, finetuned_models, model_name, file_name, mode):
    new_validation_windows, new_validation_labels = util.make_windows(validation_set, WINDOW_SIZE, HORIZON)

    new_kmeans = cluster_data(validation_set)
    clusters_no = new_kmeans.n_clusters
    cluster_labels = new_kmeans.labels_
    new_centers = new_kmeans.cluster_centers_

    clustered_windows, clustered_labels = util.create_clustered_data(
        new_validation_windows, new_validation_labels, clusters_no, cluster_labels
    )

    previous_n_clusters = kmeans.n_clusters
    previous_centers = kmeans.cluster_centers_

    cluster_mapping = {}
    matched_old_models = set()

    # Collect all similarity scores to determine an adaptive threshold
    all_distances = []
    for i in range(clusters_no):
        for j in range(previous_n_clusters):
            similarity_score = np.linalg.norm(new_centers[i] - previous_centers[j])
            all_distances.append(similarity_score)

    if all_distances:
        dynamic_threshold = np.percentile(all_distances, 30)
    else:
        dynamic_threshold = 0.5

    # Matching process using the dynamic threshold
    for i in range(clusters_no):
        best_match_idx = None
        best_similarity = float('inf')

        for j in range(previous_n_clusters):
            similarity_score = np.linalg.norm(new_centers[i] - previous_centers[j])

            if similarity_score < best_similarity:
                best_similarity = similarity_score
                best_match_idx = j

        cluster_mapping[i] = best_match_idx if best_similarity < dynamic_threshold else None

    # Step 6: Train or fine-tune models in correct order
    updated_models = [None] * clusters_no  # Ensure correct order
    base_model = get_base_model(model_name, file_name)

    for i in range(clusters_no):
        save_dir = f"/Users/aalademi/PycharmProjects/ecml/Models/{file_name}/{mode}/{model_name}/cluster{i+1}"

        cluster_windows = clustered_windows[i]
        cluster_labels = clustered_labels[i]
        matched_old_idx = cluster_mapping[i]

        # Case 1: Highly similar cluster → Fine-tune the corresponding old model
        if matched_old_idx is not None and matched_old_idx not in matched_old_models:
            print(f"Cluster {i} matches old cluster {matched_old_idx}, fine-tuning...")
            pretrained_model = copy.deepcopy(finetuned_models[matched_old_idx])
            cluster_model = models_util.train_specialized_models(pretrained_model, cluster_windows, cluster_labels, save_dir)
            matched_old_models.add(matched_old_idx)
        else:
            print(f"Cluster {i} is entirely new, initializing from the base model...")
            cluster_model = models_util.train_specialized_models(base_model, cluster_windows, cluster_labels, save_dir)

        # Store model in correct cluster order
        updated_models[i] = cluster_model

    return updated_models, new_kmeans



def detect_concept_drift(validation_set, mean_ref, hoeffding_bound, finetuned_models, kmeans, model_name, file_name):
    print("ref Mean:", mean_ref)

    # Add the new point to the validation set and update the mean

    mean_new = np.mean(validation_set[int(len(validation_set) * 0.75):])
    # mean_new = np.mean(validation_set)

    print("current mean:", mean_new)

    # Measure the deviation
    delta = abs(mean_new - mean_ref)

    print("Delta : ", delta)
    print("Hoeffding bound : ", hoeffding_bound)
    new_kmeans = None
    # Check for concept drift
    if delta > hoeffding_bound:
        print("Concept drift detected!")
        # Update the reference mean
        mean_ref = mean_new
        print("updated Mean:", mean_ref)
        print("retrigger the clustering")
        finetuned_models, new_kmeans = retrigger_clustering(kmeans, validation_set, finetuned_models, model_name,
                                                            file_name, "online")

    return mean_ref, finetuned_models, new_kmeans


def normalize_rmse(rmse, data):
    min_val, max_val = np.min(data), np.max(data)
    range_val = max_val - min_val
    return rmse / range_val if range_val != 0 else 0


def evaluate_online(finetuned_models, model_name, file_name, val, test, kmeans, original_data):
    test_windows, test_labels = util.make_windows(test, WINDOW_SIZE, HORIZON)

    mean_ref = np.mean(val[int(len(val) * 0.75):])
    hoeffding_bound = util.compute_hoeffding_bound(WINDOW_SIZE)

    cluster_centers = kmeans.cluster_centers_
    drift_points = 0
    predictions = []
    for t in range(len(test_windows)):
        recent_window = test_windows[t]
        recent_label = test_labels[t].item()

        prediction = predict_value(finetuned_models, model_name, recent_window, cluster_centers)
        predictions.append(prediction)

        val = np.concatenate((val[1:], np.array([[recent_label]])), axis=0)
        drift_points += 1
        mean_ref, finetuned_models, updated_kmeans = detect_concept_drift(val, mean_ref,
                                                                  hoeffding_bound, finetuned_models, kmeans, model_name,
                                                                  file_name,
                                                                  )
        if updated_kmeans is not None:
            kmeans = updated_kmeans
            cluster_centers = kmeans.cluster_centers_
        print(f"size of vaildation_set after : {len(val)}")
        print(f'size of models : {len(finetuned_models)}')
        print(f'size of clusters centers : {len(cluster_centers)}')

    predictions = np.array(predictions)
    labels = np.array(test_labels).squeeze()

    # Inverse-transform to the original scale.
    predictions_orig = scaler.inverse_transform(predictions.reshape(-1, 1)).squeeze()
    labels_orig = scaler.inverse_transform(labels.reshape(-1, 1)).squeeze()

    mse_orig = np.mean((predictions_orig - labels_orig) ** 2)
    rmse_orig = np.sqrt(mse_orig)

    nrmse_orig = normalize_rmse(rmse_orig, original_data)

    print("##########################")
    print(f"Predictions (original scale): {predictions_orig}")
    print("##########################")

    print(f"rmse of finetuned models: {rmse_orig}")
    print(f"nrmse of finetuned models: {nrmse_orig}")

    return rmse_orig, nrmse_orig


def evaluate_offline(finetuned_models, model_name, file_name, test, cluster_centers, original_data):
    predictions = []
    test_windows, test_labels = util.make_windows(test, WINDOW_SIZE, HORIZON)
    for t in range(len(test_windows)):
        recent_window = test_windows[t]
        prediction = predict_value(finetuned_models, model_name, recent_window, cluster_centers)
        predictions.append(prediction)

    predictions = np.array(predictions)
    labels = np.array(test_labels).squeeze()

    # Inverse-transform to the original scale.
    predictions_orig = scaler.inverse_transform(predictions.reshape(-1, 1)).squeeze()
    labels_orig = scaler.inverse_transform(labels.reshape(-1, 1)).squeeze()

    mse_orig = np.mean((predictions_orig - labels_orig) ** 2)
    rmse_orig = np.sqrt(mse_orig)

    nrmse_orig = normalize_rmse(rmse_orig, original_data)

    print("##########################")
    print(f"Predictions (original scale): {predictions_orig}")
    print("##########################")

    print(f"mse of finetuned models: {mse_orig}")
    print(f"nrmse of finetuned models: {nrmse_orig}")

    return rmse_orig, nrmse_orig

def detect_periodic_concept_drift(kmeans, validation_set, update, models, model_name, file_name):
    # Check for concept drift
    new_kmeans = None
    if update:
        models, new_kmeans = retrigger_clustering(kmeans, validation_set, models, model_name, file_name, "periodic")

    return models, new_kmeans


def compute_update_periods(test_size):
    update_intervals = math.floor(test_size * 0.10)  # Compute update interval (every 10%)
    return list(range(update_intervals, test_size, update_intervals))  # Generate update points


def evaluate_periodic(finetuned_models, model_name, file_name, val, test, kmeans, original_data):
    test_windows, test_labels = util.make_windows(test, WINDOW_SIZE, HORIZON)

    cluster_centers = kmeans.cluster_centers_
    drift_points = 0
    predictions = []
    update_periods = compute_update_periods(len(test_labels))
    for t in range(len(test_windows)):
        recent_window = test_windows[t]
        recent_label = test_labels[t].item()

        prediction = predict_value(finetuned_models, model_name, recent_window, cluster_centers)
        predictions.append(prediction)

        val = np.concatenate((val[1:], np.array([[recent_label]])), axis=0)
        drift_points += 1
        if t in update_periods:
            print(f"Triggering model update at time step {t}")
            finetuned_models, updated_kmeans = detect_periodic_concept_drift(
                kmeans, val, True, finetuned_models, model_name, file_name)
            if updated_kmeans is not None:
                kmeans = updated_kmeans
                cluster_centers = kmeans.cluster_centers_

        print(f"size of vaildation_set after : {len(val)}")
        print(f'size of models : {len(finetuned_models)}')
        print(f'size of clusters centers : {len(cluster_centers)}')

    predictions = np.array(predictions)
    labels = np.array(test_labels).squeeze()

    # Inverse-transform to the original scale.
    predictions_orig = scaler.inverse_transform(predictions.reshape(-1, 1)).squeeze()
    labels_orig = scaler.inverse_transform(labels.reshape(-1, 1)).squeeze()

    mse_orig = np.mean((predictions_orig - labels_orig) ** 2)
    rmse_orig = np.sqrt(mse_orig)
    nrmse_orig = normalize_rmse(rmse_orig, original_data)

    print("##########################")
    print(f"Predictions (original scale): {predictions_orig}")
    print("##########################")

    print(f"mse of finetuned models: {mse_orig}")
    print(f"nrmse of finetuned models: {nrmse_orig}")

    return rmse_orig, nrmse_orig

def predict_value(models, model_name, recent_subsequence, cluster_centers):
    # Reshape the recent subsequence to match the expected input shape
    current_window = recent_subsequence.reshape(1, -1)

    # Initialize variables to track the closest cluster center
    min_euclidean_distance = float('inf')
    closest_cluster_idx = None

    # Iterate over the cluster centers to find the one with the minimal Euclidean distance
    for idx, center in enumerate(cluster_centers):
        center = center.reshape(1, -1)  # Reshape the center to ensure consistent dimensions
        euclidean_distance = euclidean_distances(current_window, center)[0][0]  # Calculate Euclidean distance

        if euclidean_distance < min_euclidean_distance:
            min_euclidean_distance = euclidean_distance
            closest_cluster_idx = idx
    print("---------------------------------------------")
    print(f"Closest cluster index: {closest_cluster_idx}")

    # Use the model corresponding to the closest cluster center
    model = models[closest_cluster_idx]

    # Make a prediction for the next time step based on the current window
    prediction = models_util.evaluate(model, model_name, current_window)

    # Calculate MSE for this prediction
    # mse = mean_squared_error([test_labels[t]], [prediction])
    # print(f"Time {t}: Prediction: {prediction[0][0]}, Actual: {test_labels[t]}, MSE: {mse}")
    return prediction


def save_results_per_model(model_name, results):
    results_dir = f"/Users/aalademi/PycharmProjects/ecml/results/final_results"
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f"{model_name}.csv")

    fieldnames = ["Dataset", "Base RMSE", "Base NRMSE", "Offline RMSE", "Offline NRMSE",
                  "Online RMSE", "Online NRMSE", "Periodic RMSE", "Periodic NRMSE",
                  "Online Runtime", "Periodic Runtime"]

    # Check if the file already exists
    file_exists = os.path.exists(results_file)

    with open(results_file, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write header only if the file is newly created
        if not file_exists:
            writer.writeheader()

        # Append new rows
        writer.writerows(results)


def train_model(train, test, model_name, file_name, WINDOW_SIZE=7, HORIZON=1):
    train_windows, train_labels = util.make_windows(train, WINDOW_SIZE, HORIZON)
    test_windows, test_labels = util.make_windows(test, WINDOW_SIZE, HORIZON)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = f'/Users/aalademi/PycharmProjects/ecml/Models/{file_name}/base-model/{model_name}'
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
        model_builder = UnderOptModelBuilder(model_type=model_name, n_timesteps=WINDOW_SIZE, horizon=HORIZON)
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

    model_builder = UnderOptModelBuilder(model_type=model_name, n_timesteps=WINDOW_SIZE, horizon=HORIZON)
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


def prepare_data(data_path):
    file_names = os.listdir(data_path)

    for name in file_names:
        if name.lower().endswith('.csv'):
            dataset_name = os.path.splitext(name)[0]
            print(f"Processing dataset: {dataset_name}")

            data = pd.read_csv(os.path.join(data_path, name))

            if data.shape[1] < 2:
                print(f"Skipping {dataset_name} - Not enough columns.")
                continue

            values = data.iloc[:, 1].values
            if len(values) > 20000:
                data = data.iloc[:20000]
                print(f"Dataset {dataset_name} trimmed to 20,000 rows.")

            train, val, test = preprocess_data(data)

            for model_name in MODELS:
                model_results = []

                results_file = f"/Users/aalademi/PycharmProjects/ecml/results/final_results/{model_name}.csv"
                if os.path.exists(results_file):
                    existing_results = pd.read_csv(results_file)
                    if dataset_name in existing_results["Dataset"].values:
                        print(f"Skipping model: {model_name} for dataset: {dataset_name} (already processed)")
                        continue

                train_model(train, test, model_name, dataset_name)
                base_model = get_base_model(model_name, dataset_name)

                result_base, nrmse_base = evaluate_model(base_model, model_name, test, dataset_name, values)

                clustering_results = cluster_data(val)
                finetuned_models = finetune_models(base_model, model_name, dataset_name, val, clustering_results)
                finetuned_models = get_finetuned_models(model_name, dataset_name, clustering_results.n_clusters)

                result_offline, nrmse_offline = evaluate_offline(
                    finetuned_models, model_name, dataset_name, test, clustering_results.cluster_centers_, values
                )

                start_time = time.time()
                result_online, nrmse_online = evaluate_online(
                    finetuned_models, model_name, dataset_name, val, test, clustering_results, values
                )
                runtime_online = time.time() - start_time

                start_time_p = time.time()
                result_periodic, nrmse_periodic = evaluate_periodic(
                    finetuned_models, model_name, dataset_name, val, test, clustering_results, values
                )
                runtime_periodic = time.time() - start_time_p

                model_results.append({
                    "Dataset": dataset_name,
                    "Base RMSE": result_base,
                    "Base NRMSE": nrmse_base,
                    "Offline RMSE": result_offline,
                    "Offline NRMSE": nrmse_offline,
                    "Online RMSE": result_online,
                    "Online NRMSE": nrmse_online,
                    "Periodic RMSE": result_periodic,
                    "Periodic NRMSE": nrmse_periodic,
                    "Online Runtime": runtime_online,
                    "Periodic Runtime": runtime_periodic
                })

                save_results_per_model(model_name, model_results)
                reset_pytorch_configuration()




if __name__ == '__main__':
    test_files_path = "test"
    prepare_data(test_files_path)
