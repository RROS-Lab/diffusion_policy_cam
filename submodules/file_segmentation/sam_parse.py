import pandas as pd
import numpy as np

FRAME_RATE = 240

# Define the CSV data as a string
csv_path = "no-sync/aug14/trimmed_traj_with_helmet_meters/segmentation_info/sam_segment.csv"

# Read the CSV data into a pandas DataFrame
df = pd.read_csv(csv_path)

# Drop rows with any NaN values
df_clean = df.dropna(subset=df.columns.difference(['Comment']))

# Convert 'take' and 'edge' columns to integer type
df_clean['take'] = df_clean['take'].astype(int)
df_clean['edge'] = df_clean['edge'].astype(int)

# Function to convert 'HH:MMM' to float 'H.MMM'
def convert_time_format(value):
    if pd.isna(value):
        return None
    HH, MM, SS, FRAMES = value.split(':')
    # SS, MS = value.split(':')
    TOTAL_FRAMES = round(int(SS) * 240 + int(FRAMES))
    return TOTAL_FRAMES


# Apply the conversion to 'S1', 'E1', 'S2', 'E2', 'S3', and 'E3' columns
columns_to_convert = ['S1', 'E1', 'S2', 'E2', 'S3', 'E3']
for column in columns_to_convert:
    df_clean[column] = df_clean[column].apply(convert_time_format)

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
        new_data.append([take, next_edge, start, end, step, comment])
        next_edge = (start_edge + i) % 4 + 1  # Rotate edges (1 -> 2 -> 3 -> 4 -> 1)

# Create the new DataFrame
df_transformed = pd.DataFrame(new_data, columns=['take', 'edge', 'start', 'end', 'step', 'comment'])

df_transformed['take'] = df_transformed['take'].astype(int)
df_transformed['step'] = df_transformed['step'].astype(int)
df_transformed['edge'] = df_transformed['edge'].astype(int)
# Display the new DataFrame
print(df_transformed)

df_transformed.to_csv("no-sync/aug14/trimmed_traj_with_helmet_meters/segmentation_info/sam_segment_transformed.csv", index=False)