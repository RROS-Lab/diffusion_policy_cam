import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# from ..submodules import data_filter as _df
# import submodules.robomath as rm
# import submodules.robomath_addon as rma
# import submodules.motive_file_cleaner as mfc
import submodules.data_filter as _df
# import pandas as pd


print("..........")

test_file = "/home/cam/Documents/raj/diffusion_policy_cam/no-sync/turn_table_chisel/tilt_25/1.cleaned_data/training_traj/cap_001_cleaned.csv"
data = pd.read_csv(test_file)

print(data.head())

data = _df.apply_savgol_filter(data, window_size=5, polyorder=3)

print(data.head())