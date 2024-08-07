import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)


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

cross_ref_limit = 5
Body_type = 'rb'
tolerance_sheet = [0.02, 0.02, 0.02]
tolerance_gripper = [0.005, 0.06, 0.005]

_params = {
    'RigidBody': {'len':7,
                'dof': ['X', 'Y', 'Z', 'w', 'x', 'y', 'z']},
    'Marker': {'len':3,
                'dof': ['X', 'Y', 'Z']}
}


gripper_marker_name = ['GS']
RigidBody_OI = ['battery', 'chisel', 'gripper']
REF_FRAME = 100


dir_path = '/home/cam/Documents/raj/diffusion_policy_cam/no-sync/turn_table_chisel/tilt_25/dataset_Aug_6/raw_traj/'
save_path = '/home/cam/Documents/raj/diffusion_policy_cam/no-sync/turn_table_chisel/tilt_25/dataset_Aug_6/cleaned_traj/'

B_MOIs = mfc._get_sheet_marker_limit(dir_path, RigidBody_OI ,Body_type, 'battery', REF_FRAME, tolerance_sheet, cross_ref_limit)
# G_MOIs = mfc._get_marker_limit(dir_path, RigidBody_OI ,Body_type, 'gripper', REF_FRAME, tolerance_gripper, gripper_marker_name, cross_ref_limit)


MOIs = {'battery': B_MOIs}
print(len(MOIs['battery']['pos']))

Oi = {'RigidBody': RigidBody_OI, 'Marker': B_MOIs['names']}
files = os.listdir(dir_path)


for file in files:

    csv_path = os.path.join(dir_path, file)
    save_file = re.sub(r'\.csv', '_cleaned.csv', file)
    save_file_path = os.path.join(save_path, save_file)

    mfc.motive_chizel_task_cleaner(csv_path=csv_path, save_path=save_file_path, OI = Oi, _params = _params, REF_FRAME = REF_FRAME, MOIs = MOIs)