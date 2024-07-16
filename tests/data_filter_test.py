import submodules.robomath as rm
import submodules.robomath_addon as rma
import submodules.motive_file_cleaner as mfc
import submodules.data_filter as _df
import pandas as pd


print("..........")

test_file = "test.csv"
data = pd.read_csv(test_file)

print(data.head())

data = _df.apply_savgol_filter(data, window_size=5, polyorder=3)

print(data.head())