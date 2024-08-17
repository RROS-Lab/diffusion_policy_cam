import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)


import submodules.cleaned_file_parser as cfp
import submodules.data_filter as _df
import numpy as np
from matplotlib import pyplot as plt
import submodules.robomath_addon as rma
import submodules.robomath as rm
import submodules.rs_rotation_index as rri
import torch.nn as nn
import torch

path = '/home/cam/Documents/raj/diffusion_policy_cam/no-sync/turn_table_chisel/tilt_25/1.cleaned_data/training_traj/cap_002_cleaned.csv'

data = cfp.DataParser.from_quat_file(file_path = path, target_fps= 120, filter=False, window_size=5, polyorder=3)
START_FRAME = 0
battery = data.get_rigid_TxyzRxyz()['battery']
battery0 = battery[0]
stop_index = rri.get_battery_stop_index(data, START_FRAME, thresehold=0.1)
marker = data.get_marker_Txyz()['A1']
start_marker = marker[0]
loss = []

for T ,(markT, batT) in enumerate(zip(marker[:stop_index], battery[:stop_index])):
    if T > 0:
        cal_markT = rri.calculate_new_marker_pos(start_marker, battery0, batT)
        loss.append(nn.functional.mse_loss(torch.tensor(cal_markT), torch.tensor(markT)))

print(len(loss))