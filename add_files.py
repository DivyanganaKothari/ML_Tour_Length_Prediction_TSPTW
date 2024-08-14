# add_files.py

import os
import pandas as pd

def main():
    # Directory containing the processed files
    individual_output_dir = 'data_ml/ProcessedFiles3'  # Path where individual processed files are saved

    # Get a list of all processed CSV files
    all_processed_files = [os.path.join(individual_output_dir, f) for f in os.listdir(individual_output_dir) if f.endswith('.csv')]

    # Define the path for the final combined output
    combined_output_path = 'data_ml/input_features/combined_input_features_3.csv'

    # Initialize a flag to check if it's the first file
    first_file = True

    # Iterate through all the processed files
    for file in all_processed_files:
        if first_file:
            # For the first file, include the header (column names)
            df = pd.read_csv(file)
            df.to_csv(combined_output_path, mode='w', index=False)
            first_file = False  # After processing the first file, set the flag to False
        else:
            # For subsequent files, skip the header and append only the data
            df = pd.read_csv(file, skiprows=1, header=None)
            df.to_csv(combined_output_path, mode='a', index=False, header=False)

    print(f"All individual files combined into {combined_output_path}")

if __name__ == "__main__":
    main()
