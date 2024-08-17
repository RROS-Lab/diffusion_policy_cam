import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)


from get_video_from_csv import get_video_from_csv
import os
from tqdm import tqdm  # Import tqdm for progress bar functionality

SOURCE_DIR = './../datasets/jul15_prediction/cleaned_test_data'
TARGET_DIR = './saved_animations/cleaned_test_data'

# Get all files in the source directory and filter out the csv files
input_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.csv')]

# Use tqdm to create a progress bar for the loop
for file in tqdm(input_files, desc='Processing CSV Files', unit='file'):
    source_path = os.path.join(SOURCE_DIR, file)
    target_path = os.path.join(TARGET_DIR, file.replace('.csv', '.mp4'))
    get_video_from_csv(source_path=source_path, target_path=target_path)

print('Done!')
