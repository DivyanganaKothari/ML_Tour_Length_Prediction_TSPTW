import pandas as pd

#manually change csv to excel and format the file as per standard for this code
#file need to be in excel


# Load the data
shipment_data = pd.read_excel('raw_files/Depot-521/20240403_521_shipments.xlsx')
distance_matrix = pd.read_excel('raw_files/Depot-521/20240403_521_distanceMatrix.xlsx', index_col=0)


# Step 1: Filter the shipment entries to retain unique address IDs
unique_shipments = shipment_data.drop_duplicates(subset='AddressId').copy()

# Map the unique address IDs to sequential integers using .loc
unique_address_ids = unique_shipments['AddressId']

# Step 2: Filter the distance matrix to only include rows and columns with unique address IDs
filtered_distance_matrix = distance_matrix.loc[unique_address_ids, unique_address_ids]

## Save the filtered distance matrix
filtered_distance_matrix.to_csv('Data/Depot-521/filtered_distance_matrix.csv')

# Save the filtered shipment entries
unique_shipments.to_csv('Data/Depot-521/filtered_shipment_entries.csv', index=False)

