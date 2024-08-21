import pandas as pd

# Define the path to the transformed CSV file
transformed_csv_path = "no-sync/aug14/trimmed_traj_with_helmet_meters/segmentation_info/sam_segment_2_transformed.csv"

# Read the transformed CSV data into a pandas DataFrame
df_transformed = pd.read_csv(transformed_csv_path)

# Display the DataFrame to verify the content
print(df_transformed)