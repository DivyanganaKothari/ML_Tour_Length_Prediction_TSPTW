import pandas as pd
import numpy as np
import os
import ast
import add_files

# Load the filtered shipment entries for Depot 521
filtered_shipment_entries = pd.read_csv('Data/Depot-521/filtered_shipment_entries.csv')

# Load the distance matrix for Depot 521
distance_matrix = pd.read_csv('Data/Depot-521/filtered_distance_matrix.csv', index_col=0)

# Ensure AddressId columns are treated as integers
distance_matrix.index = distance_matrix.index.astype(int)
distance_matrix.columns = distance_matrix.columns.astype(int)

# Load the depot node information for Depot 521
depot_node_info = pd.DataFrame([{
    'AddressId': 9,
    'Von1': '03-04-2024 00:00:00',
    'Bis1': '03-04-2024 23:59:59',
    'Latitude': 46.02154,
    'Longitude': 8.91833
}])

# Ensure depot AddressId 9 is added to the distance matrix
if 9 not in distance_matrix.index:
    depot_distances = pd.Series(0, index=distance_matrix.columns)
    distance_matrix.loc[9] = depot_distances
    distance_matrix[9] = depot_distances

# Directory containing the zip code files for Depot 521
zip_code_files_dir = 'zipCode_Depot_521'

# Directory to save the individual processed files for Depot 521
individual_output_dir = 'data_ml/ProcessedFiles521'
if not os.path.exists(individual_output_dir):
    os.makedirs(individual_output_dir)

# Function to calculate time window features
def calculate_time_window_features(filtered_info):
    filtered_info['Von1'] = pd.to_datetime(filtered_info['Von1'], format='mixed', dayfirst=True, errors='coerce')
    filtered_info['Bis1'] = pd.to_datetime(filtered_info['Bis1'], format='mixed', dayfirst=True, errors='coerce')


    mask = ~((filtered_info['Von1'] == pd.Timestamp('03-04-2024 00:00:00')) &
             (filtered_info['Bis1'] == pd.Timestamp('03-04-2024 23:59:59')))
    calculation_info = filtered_info[mask]

    reference_time = pd.Timestamp('03-04-2024 00:00:00')  # Adjusted to match depot start time

    time_windows = [(row['Von1'], row['Bis1']) for _, row in calculation_info.iterrows()]
    if len(time_windows) == 0:
        return {
            'Total Time Window': -1,
            'Average Time Window': -1,
            'Standard Deviation of Time Window': -1,
            'Average Earliest Time': -1,
            'Standard Deviation Earliest Time': -1,
            'Average Latest Time': -1,
            'Standard Deviation Latest Time': -1,
            'Mean Time Window': -1
        }

    total_time_window = sum([(end - start).total_seconds() / 60 for start, end in time_windows])
    average_time_window = total_time_window / len(time_windows)
    std_dev_time_window = np.std([(end - start).total_seconds() / 60 for start, end in time_windows])

    earliest_times = [(tw[0] - reference_time).total_seconds() / 60 for tw in time_windows]
    min_earliest_time = np.min(earliest_times)
    max_earliest_time = np.max(earliest_times)
    average_earliest_time = np.mean(earliest_times)
    std_dev_earliest_time = np.std(earliest_times)

    latest_times = [(tw[1] - reference_time).total_seconds() / 60 for tw in time_windows]
    min_latest_time = np.min(latest_times)
    max_latest_time = np.max(latest_times)
    average_latest_time = np.mean(latest_times)
    std_dev_latest_time = np.std(latest_times)

    mean_time_window = np.mean([(end - start).total_seconds() for start, end in time_windows])

    return {
        'Total Time Window': total_time_window,
        'Average Time Window': average_time_window,
        'Standard Deviation of Time Window': std_dev_time_window,
        'Average Earliest Time': average_earliest_time,
        'Standard Deviation Earliest Time': std_dev_earliest_time,
        'Average Latest Time': average_latest_time,
        'Standard Deviation Latest Time': std_dev_latest_time,
        'Min Earliest Time': min_earliest_time,
        'Max Earliest Time': max_earliest_time,
        'Min Latest Time': min_latest_time,
        'Max Latest Time': max_latest_time,
        'Mean Time Window': mean_time_window
    }

# Function to calculate distance features
def calculate_distance_features(filtered_info, distance_matrix):
    address_ids = filtered_info['AddressId'].astype(int).tolist()
    if 9 not in address_ids:
        address_ids.append(9)

    distance_matrix_cluster = distance_matrix.loc[address_ids, address_ids]
    n = distance_matrix_cluster.shape[0]
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    all_distances = distance_matrix_cluster.values[mask]

    avg_node = np.mean(distance_matrix_cluster.values, axis=0)
    euclidean_distances_to_avg_node = np.linalg.norm(distance_matrix_cluster.values - avg_node[:, np.newaxis], axis=1)

    nearest_neighbors = np.min(distance_matrix_cluster.values + np.eye(n) * np.max(distance_matrix_cluster.values), axis=1)
    farthest_neighbors = np.max(distance_matrix_cluster.values, axis=1)

    return {
        'Total Number of Nodes': n,
        'MinP': np.min(all_distances),
        'MaxP': np.max(all_distances),
        'VarP': np.var(all_distances),
        'SumMinP': np.sum(nearest_neighbors),
        'SumMaxP': np.sum(farthest_neighbors),
        'MinM': np.min(euclidean_distances_to_avg_node),
        'MaxM': np.max(euclidean_distances_to_avg_node),
        'SumM': np.sum(euclidean_distances_to_avg_node),
        'VarM': np.var(euclidean_distances_to_avg_node),
        'Mean Distance': np.mean(all_distances),
        'Median Distance': np.median(all_distances),
        'Std Distance': np.std(all_distances),
        'Sum Distance': np.sum(all_distances),
        'Sum of Min Distance': np.sum(np.min(all_distances)),
        'Sum of Max Distance': np.sum(np.max(all_distances)),
        'Percentile 25 Distance': np.percentile(all_distances, 25),
        'Percentile 50 Distance': np.percentile(all_distances, 50),
        'Percentile 75 Distance': np.percentile(all_distances, 75)
    }
def calculate_depot_features(filtered_info, distance_matrix):
    depot_distances = distance_matrix.loc[9, filtered_info['AddressId'].astype(int).tolist()].values
    depot_distances_excluding_self = depot_distances[depot_distances != 0]

    return {
        'Sum of Distance to Depot': np.sum(depot_distances_excluding_self),
        'Average Distance to Depot': np.mean(depot_distances_excluding_self),
        'Maximum Distance to Depot': np.max(depot_distances_excluding_self),
        'Minimum Distance to Depot': np.min(depot_distances_excluding_self),
        'Standard Deviation of Distance to Depot': np.std(depot_distances_excluding_self)
    }

def calculate_input_features(filtered_info, distance_matrix, cluster_id, tour_length):
    distance_features = calculate_distance_features(filtered_info, distance_matrix)
    time_window_features = calculate_time_window_features(filtered_info)
    depot_features = calculate_depot_features(filtered_info, distance_matrix)

    features = {
        **distance_features,
        **time_window_features,
        **depot_features,
        'Tour Length': tour_length
    }
    return features


# Iterate over each zip code file and process individually
for zip_code_file in os.listdir(zip_code_files_dir):
    if zip_code_file.endswith('.csv'):
        summary_info = []

        zip_codes_df = pd.read_csv(os.path.join(zip_code_files_dir, zip_code_file), sep=';')
        zip_codes_df['Zip Codes'] = zip_codes_df['Zip Codes'].apply(lambda x: ast.literal_eval(x.strip()))

        for index, row in zip_codes_df.iterrows():
            cluster_id = index + 1  # Assign a unique cluster ID
            zip_codes = row['Zip Codes']
            tour_length = row['Tour Length [min]']

            if tour_length == -1:
                print(f"Skipping cluster {cluster_id} in file {zip_code_file} with zip codes {zip_codes} due to invalid tour length.")
                continue

            # Filter the shipment entries based on the zip codes
            filtered_info = filtered_shipment_entries[filtered_shipment_entries['PLZ'].isin(zip_codes)]

            if not filtered_info.empty:
                # Add the depot information to the filtered info
                filtered_info = pd.concat([filtered_info, depot_node_info], ignore_index=True)
                # Calculate the input features
                try:
                    input_features = calculate_input_features(filtered_info, distance_matrix, cluster_id, tour_length)
                    summary_info.append(input_features)
                except Exception as e:
                    print(f"Error for cluster {cluster_id} in file {zip_code_file} with zip codes {zip_codes}: {e}")
            else:
                print(f"No shipment data found for cluster {cluster_id} in file {zip_code_file} with zip codes {zip_codes}")

        # Save the summary information for the current file
        output_file_path = os.path.join(individual_output_dir, f'processed_{zip_code_file}')
        summary_df = pd.DataFrame(summary_info)
        summary_df.to_csv(output_file_path, index=False)
        print(f"Processed data saved to {output_file_path}")

add_files.main()
