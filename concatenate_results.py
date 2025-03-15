import os
import pandas as pd

# Directory containing the CSV files
directory = "/Users/aalademi/PycharmProjects/ecml/results/final_results/optimized_versions/average_results"

# Output file name
output_file = "results/final_results/ranking/final_concatenated_file.csv"

# List to store DataFrames
dataframes = []

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):  # Process only CSV files
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)  # Read the CSV file
        dataframes.append(df)

# Concatenate all dataframes
if dataframes:
    final_df = pd.concat(dataframes, ignore_index=True)
    final_df.to_csv(output_file, index=False)  # Save to the final CSV file
    print(f"Final file saved as {output_file}")
else:
    print("No CSV files found in the directory.")
