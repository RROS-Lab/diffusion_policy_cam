import math
import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
# from scipy.spatial.transform import Rotation
from matplotlib.backends.backend_pdf import PdfPages
import submodules.robomath as rm
import submodules.robomath_addon as rma
import submodules.motive_file_cleaner as mfc
import warnings
warnings.filterwarnings("ignore")

print("..........")

# dir_path = '/home/cam/Documents/diffusion_policy_cam/diffusion_pipline/data_chisel_task/raw_traj/'
# save_path = '/home/cam/Documents/diffusion_policy_cam/diffusion_pipline/data_chisel_task/cleaned_traj/'

# for file in os.listdir(dir_path):
#     if file.endswith(".csv"):
#         path = os.path.join(dir_path, file)
#         # print(path)
#         mfc.motive_chizel_task_cleaner(
#             csv_path = path, save_path=save_path
#         )

path = '/home/cam/Documents/diffusion_policy_cam/diffusion_pipline/data_chisel_task/test_128_raw.csv'
save_path = '/home/cam/Documents/diffusion_policy_cam/diffusion_pipline/data_chisel_task/'

mfc.motive_chizel_task_cleaner(csv_path=path, save_path=save_path)