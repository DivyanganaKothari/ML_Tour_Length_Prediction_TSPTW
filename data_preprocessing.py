import pandas as pd

# Load the data
shipment_data = pd.read_excel('raw_files/shipments_entries.xlsx')
distance_matrix = pd.read_excel('raw_files/distanceMatrix.xlsx', index_col=0)

# Step 1: Filter the shipment entries to retain unique address IDs
unique_shipments = shipment_data.drop_duplicates(subset='AddressId').copy()

# Map the unique address IDs to sequential integers using .loc
unique_address_ids = unique_shipments['AddressId']

# Step 2: Filter the distance matrix to only include rows and columns with unique address IDs
filtered_distance_matrix = distance_matrix.loc[unique_address_ids, unique_address_ids]

## Save the filtered distance matrix
filtered_distance_matrix.to_csv('Data/filtered_distance_matrix.csv')

# Save the filtered shipment entries
unique_shipments.to_csv('Data/filtered_shipment_entries.csv', index=False)

