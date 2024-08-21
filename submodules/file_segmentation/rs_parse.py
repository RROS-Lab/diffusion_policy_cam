import pandas as pd
import numpy as np

# Define the CSV data as a string
csv_path = "no-sync/aug14/trimmed_traj_with_helmet_meters/segmentation_info/raj_segment.csv"

# Read the CSV data into a pandas DataFrame
df = pd.read_csv(csv_path)

# Drop rows with any NaN values
df_clean = df.dropna(subset=df.columns.difference(['Comment']))

# Convert 'take' and 'edge' columns to integer type
df_clean['take'] = df_clean['take'].astype(int)
df_clean['edge'] = df_clean['edge'].astype(int)

# Apply the conversion to 'S1', 'E1', 'S2', 'E2', 'S3', and 'E3' columns
columns_to_convert_to_float = ['S1', 'E1', 'S2', 'E2', 'S3', 'E3']
for column in columns_to_convert_to_float:
    df_clean[column] = df_clean[column].astype(float)

# Create a new DataFrame for the desired format
new_data = []

for _, row in df_clean.iterrows():
    take = row['take']
    start_edge = row['edge']
    comment = row['Comment']
    
    steps = [
        (row['S1'], row['E1'], 1),
        (row['S2'], row['E2'], 2),
        (row['S3'], row['E3'], 3)
    ]
    
    next_edge = start_edge
    for i, (start, end, step) in enumerate(steps):
        new_data.append([take, next_edge, round(start*240), round(end*240), step, comment])
        next_edge = (start_edge + i) % 4 + 1  # Rotate edges (1 -> 2 -> 3 -> 4 -> 1)

# Create the new DataFrame
df_transformed = pd.DataFrame(new_data, columns=['take', 'edge', 'start', 'end', 'step', 'comment'])

df_transformed['take'] = df_transformed['take'].astype(int)
df_transformed['step'] = df_transformed['step'].astype(int)
df_transformed['edge'] = df_transformed['edge'].astype(int)

# Display the new DataFrame
print(df_transformed)

df_transformed.to_csv("no-sync/aug14/trimmed_traj_with_helmet_meters/segmentation_info/raj_segment_transformed.csv", index=False)