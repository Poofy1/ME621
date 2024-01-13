import pandas as pd
import os

# Paths
csv_file_path = 'K:/Temp_SSD_Data/images_300.csv'   # Replace with the path to your CSV file
new_base_path = 'K:/Temp_SSD_Data/images_300/'

# Read the CSV file
df = pd.read_csv(csv_file_path, header=None, names=['old_path'])

# Convert paths
df['new_path'] = df['old_path'].apply(lambda x: os.path.join(new_base_path, os.path.basename(x)))

# Save the new paths to a new CSV file
new_csv_file_path = 'K:/Temp_SSD_Data/images_300.csv'  # Replace with the desired path for the new CSV file
df['new_path'].to_csv(new_csv_file_path, index=False, header=False)
