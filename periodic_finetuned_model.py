import copy
import csv
import math
import random
import os
import time

import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.preprocessing import StandardScaler


import models_util
import util
from models_config import ModelBuilder
from models_util import load_model
from util import perform_clustering

HORIZON = 1
WINDOW_SIZE = 7
BATCH_SIZE = 128
EPOCHS = 50
LR = 0.001
SEED = 42

MODELS = ModelBuilder.get_available_models()
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
        save_dir = f"/Init_Models/{dataset}/offline/{model_name}/cluster{i + 1}"
        cluster_windows = clustered_windows[i]
        cluster_labels = clustered_labels[i]
        cluster_model = models_util.train_specialized_models(base_model, cluster_windows, cluster_labels, save_dir)
        finetuned_models.append(cluster_model)

    return finetuned_models


def get_base_model(model_name, dataset_name):
    path = f"/Init_Models/{dataset_name}/base-model/{model_name}"
    model = load_model(model_name, path)
    return model


def get_finetuned_models(model_name, dataset_name, clusters_no):
    finetuned_models = []
    for i in range(clusters_no):
        path = f"/Init_Models/{dataset_name}/offline/{model_name}/cluster{i + 1}"
        model = load_model(model_name, path)
        finetuned_models.append(model)

    return finetuned_models


def average_model_weights(models):
    averaged_weights = []
    for weights in zip(*[model.get_weights() for model in models]):
        averaged_weights.append(np.mean(weights, axis=0))

    return averaged_weights


def fuse_clusters(kmeans_old, kmeans_new, threshold=0.1):
    """
    Fuse similar clusters based on Euclidean distance between cluster centers.

    Args:
        kmeans_old: The existing KMeans model before retriggering clustering.
        kmeans_new: The new KMeans model obtained after retriggering clustering.
        threshold: Distance threshold for considering two clusters as similar.

    Returns:
        A new KMeans model with merged clusters and updated labels.
    """

    # Get cluster centers and labels
    old_centers = kmeans_old.cluster_centers_
    new_centers = kmeans_new.cluster_centers_
    old_labels = kmeans_old.labels_
    new_labels = kmeans_new.labels_

    # Compute pairwise distances between old and new cluster centers
    distance_matrix = pairwise_distances(old_centers, new_centers, metric='euclidean')

    merged_clusters = list(old_centers)
    label_mapping = {}

    for i, new_center in enumerate(new_centers):
        # Find the closest existing cluster
        min_dist = np.min(distance_matrix[:, i])
        closest_old_idx = np.argmin(distance_matrix[:, i])

        if min_dist < threshold:
            # Merge new cluster with the closest old cluster by averaging
            merged_clusters[closest_old_idx] = (merged_clusters[closest_old_idx] + new_center) / 2
            label_mapping[i] = closest_old_idx
        else:
            # Add as a new separate cluster
            merged_clusters.append(new_center)
            label_mapping[i] = len(merged_clusters) - 1

    # Update labels based on the mapping
    merged_labels = np.array([label_mapping[label] for label in new_labels])

    # Convert merged clusters back to a KMeans model
    merged_kmeans = KMeans(n_clusters=len(merged_clusters), random_state=42)
    merged_kmeans.cluster_centers_ = np.array(merged_clusters)
    merged_kmeans.labels_ = merged_labels

    return merged_kmeans


def retrigger_clustering(kmeans, validation_set, finetuned_models, model_name, file_name):
    """
    Re-performs clustering and updates machine learning models by fine-tuning or initializing new models.

    Args:
        kmeans: Previous KMeans model.
        validation_set: New validation data.
        finetuned_models: List of previously fine-tuned models (one per old cluster).
        model_name: Name of the base model.
        file_name: File name identifier for storing models.

    Returns:
        updated_models: List of updated models in the correct order.
        new_kmeans: Newly trained KMeans model.
    """

    # Step 1: Create new validation windows and labels
    new_validation_windows, new_validation_labels = util.make_windows(validation_set, WINDOW_SIZE, HORIZON)

    # Step 2: Perform new clustering
    new_kmeans = cluster_data(validation_set)
    clusters_no = new_kmeans.n_clusters
    cluster_labels = new_kmeans.labels_
    new_centers = new_kmeans.cluster_centers_

    # Step 3: Create clustered datasets
    clustered_windows, clustered_labels = util.create_clustered_data(
        new_validation_windows, new_validation_labels, clusters_no, cluster_labels
    )

    # Step 4: Get previous clusters and models
    previous_n_clusters = kmeans.n_clusters
    previous_centers = kmeans.cluster_centers_

    # Step 5: Map new clusters to old clusters based on minimum Euclidean distance
    cluster_mapping = {}  # Maps new cluster index -> best matching old cluster index
    matched_old_models = set()  # Track which old models have been reused

    for i in range(clusters_no):
        best_match_idx = None
        best_similarity = float('inf')  # Lower is better (Euclidean distance)

        for j in range(previous_n_clusters):
            similarity_score = np.linalg.norm(new_centers[i] - previous_centers[j])  # Euclidean distance

            if similarity_score < best_similarity:
                best_similarity = similarity_score
                best_match_idx = j

        cluster_mapping[i] = best_match_idx if best_similarity < 0.5 else None  # Map only if similarity is close

    # Step 6: Train or fine-tune models in correct order
    updated_models = [None] * clusters_no  # Ensure correct order
    base_model = get_base_model(model_name, file_name)

    for i in range(clusters_no):
        save_dir = f"/Init_Models/{file_name}/online/{model_name}/cluster{i}"

        cluster_windows = clustered_windows[i]
        cluster_labels = clustered_labels[i]
        matched_old_idx = cluster_mapping[i]

        # Case 1: Highly similar cluster → Fine-tune the corresponding old model
        if matched_old_idx is not None and matched_old_idx not in matched_old_models:
            print(f"Cluster {i} matches old cluster {matched_old_idx}, fine-tuning...")
            pretrained_model = copy.deepcopy(finetuned_models[matched_old_idx])
            cluster_model = models_util.train_specialized_models(pretrained_model, cluster_windows, cluster_labels,
                                                                 save_dir)
            matched_old_models.add(matched_old_idx)
        else:
            print(f"Cluster {i} is entirely new, initializing from the base model...")
            cluster_model = models_util.train_specialized_models(base_model, cluster_windows, cluster_labels, save_dir)

        # Store model in correct cluster order
        updated_models[i] = cluster_model

    return updated_models, new_kmeans


def detect_periodic_concept_drift(kmeans, validation_set, update, models, model_name, file_name):
    # Check for concept drift
    new_kmeans = None
    if update:
        models, new_kmeans = retrigger_clustering(kmeans, validation_set, models, model_name, file_name)

    return models, new_kmeans


def compute_update_periods(test_size):
    update_intervals = math.floor(test_size * 0.10)  # Compute update interval (every 10%)
    return list(range(update_intervals, test_size, update_intervals))  # Generate update points


def evaluate_periodic(finetuned_models, model_name, file_name, val, test, kmeans):
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
            finetuned_models, updated_kmeans = detect_concept_drift(
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

    print("##########################")
    print(f"Predictions (original scale): {predictions_orig}")
    print("##########################")

    print(f"mse of finetuned models: {mse_orig}")

    return rmse_orig


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


def save_results_to_csv(dataset_name, results):
    """
    Update an existing CSV file by adding the 'Periodic' and 'Periodic Runtime' columns.

    Args:
        dataset_name (str): Name of the dataset.
        results (list of dicts): List containing model evaluation results.
    """
    results_dir = f"/Users/aalademi/PycharmProjects/ecml/results/evaluation/datasets2"
    results_file = os.path.join(results_dir, f"{dataset_name}.csv")

    # Load the existing file
    if os.path.exists(results_file):
        df_existing = pd.read_csv(results_file)

        # Create a mapping from results list
        results_dict = {entry["Model"]: (entry["Periodic"], entry["Periodic Runtime"]) for entry in results}

        # Add new columns
        df_existing["Periodic"] = df_existing["Model"].map(lambda x: results_dict.get(x, (None, None))[0])
        df_existing["Periodic Runtime"] = df_existing["Model"].map(lambda x: results_dict.get(x, (None, None))[1])

        # Save back to the same file
        df_existing.to_csv(results_file, index=False)
        print(f"Updated results saved to {results_file}")

    else:
        print(f"File {results_file} not found. Creating a new one.")
        fieldnames = ["Model", "Periodic", "Periodic Runtime"]
        with open(results_file, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)


def prepare_data(data_path):
    file_names = os.listdir(data_path)

    for name in file_names:
        if name.lower().endswith('.csv'):
            dataset_name = os.path.splitext(name)[0]
            print("Processing dataset:", dataset_name)

            data = pd.read_csv(os.path.join(data_path, name))

            if data.shape[1] < 2:
                print(f"Skipping {dataset_name} - Not enough columns.")
                continue

            values = data.iloc[:, 1].values

            if len(values) > 20000:
                data = data.iloc[:20000]
                print(f"Dataset {dataset_name} has more than 20000 rows. Selected the first 20000 rows.")

            print(f"Data shape for {dataset_name} after processing: {data.shape}")

            train, val, test = preprocess_data(data)

            dataset_results = []  # Store results for this dataset

            for model_name in MODELS:
                # models_util.train_model(train, test, model_name, dataset_name)
                base_model = get_base_model(model_name, dataset_name)

                # Evaluate Base Model
                # result_base = evaluate_model(base_model, model_name, test, dataset_name)

                # Perform clustering and fine-tune models
                clustering_results = cluster_data(val)
                # finetuned_models = finetune_models(base_model, model_name, dataset_name, val, clustering_results)
                finetuned_models = get_finetuned_models(model_name, dataset_name, clustering_results.n_clusters)
                # result_offline = evaluate_offline(finetuned_models, model_name, dataset_name, test, clustering_results.cluster_centers_)
                # Evaluate Fine-tuned Init_Models
                start_time = time.time()
                result_online = evaluate_periodic(finetuned_models, model_name, dataset_name, val, test,
                                                  clustering_results)
                end_time = time.time()
                runtime = end_time - start_time

                # Store results for this model
                dataset_results.append({
                    "Model": model_name,
                    "Periodic": result_online,
                    "Periodic Runtime": runtime
                })

            save_results_to_csv(dataset_name, dataset_results)

            reset_pytorch_configuration()


if __name__ == '__main__':
    test_files_path = "test3"
    prepare_data(test_files_path)
