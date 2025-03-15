# Online Adaptive Finetuned Forecasting Framework

## Description

The **Online Adaptive Fine-tuned (OAF) forecasting framework** is developed to address the challenges posed by the dynamic nature and time-dependent patterns of time series data. By combining the strengths of deep neural networks (DNNs) with real-time adaptability, the framework provides a robust solution to handle evolving patterns and non-stationary characteristics inherent in time series forecasting.

## Project Structure

The project contains the following key directories:

- **test-files**: Includes the data needed for the experiment.
- **model\_config**: Contains the configuration of parameters and architectures of the models.
- **underopt\_models\_config**: Includes the configuration of parameters and architectures of the underoptimized models.

## Running the Experiment

### **Prerequisites**

Ensure that the required dependencies are installed and directories are correctly adjusted before running the scripts.

### **Execution Steps**

1. Run the script **`online_finetuned_model.py`** to train and evaluate the models in four versions: **Base, Offline, Online, and Periodic**.
2. Run the script **`under_opt_finetuned.py`** to train and evaluate the underoptimized models in four versions: **Base, Offline, Online, and Periodic**.
3. Run **`evaluate_models.py`** to compute the average performance results of each model across all datasets.
4. Run **`rank_models.py`** to compute the global ranking of models across all datasets.
5. Run **`compute_runtime.py`** to compute the average runtime of the **Online** and **Periodic** versions of all models across all datasets.

## Contact

For further inquiries or contributions, please reach out via GitHub Issues or email the project maintainer.

---

This README provides an overview of the project structure and execution steps. Feel free to modify it according to any updates or additional details in your project.

