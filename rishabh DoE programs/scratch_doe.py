import submodules.robomath as rm
import submodules.robomath_addon as rma
import submodules.motive_file_parser as mfp
import submodules.data_filter as df
import pandas as pd


print("..........")

test_file = "test.csv"

data = df.read_cleaned_data(test_file, fps=30.0)


data = df.apply_savgol_filter(data, window_size=5, polyorder=3)