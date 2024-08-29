import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import submodules.cleaned_file_parser as cfp
import submodules.ati_file_parser as afp
import submodules.data_filter as _df
import numpy as np
import pandas as pd
import re
import csv


        
if __name__ == "__main__":

    segment_file = '/home/cam/Documents/raj/diffusion_policy_cam/no-sync/turn_table_chisel/dataset_aug14/combined_segments.csv'
    csv_path = ''
    segmet_data = pd.read_csv(segment_file)

    take =  ((csv_path.split('/')[-1]).split('_')[-1]).split('.')[0]

    take_data = segmet_data[segmet_data['take'] == int(take)]

    for i in range(len(take_data)):
        
        start = take_data.iloc[i]['start']
        end = take_data.iloc[i]['end']
        step = take_data.iloc[i]['step']
        edge = take_data.iloc[i]['edge']
        
        save_file = (re.sub(r'\.csv', f'_edge_{edge}_step_{step}.csv', csv_path)).split('/')[-1]
        state_data = cfp.DataParser.from_euler_file(file_path = csv_path, target_fps= 120, filter=False, window_size=5, polyorder=3)
        state_data.data = state_data.data.iloc[start:end]
        