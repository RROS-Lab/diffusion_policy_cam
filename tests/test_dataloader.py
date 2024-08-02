# diffusion policy import
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import pandas as pd
import math
import time
import torch
import torch.nn as nn
import collections
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import submodules.data_filter as _df
import diffusion_pipline.data_processing as dproc
import diffusion_pipline.model as md
import submodules.cleaned_file_parser as cfp



#@markdown ### **Network Demo**

# observation and action dimensions corrsponding to
# the output of PushTEnv
# obs_dim = 25
# action_dim = 13

obs_dim = 25
action_dim = 13
# parameters
pred_horizon = 16
obs_horizon = 2
action_horizon = 8
target_fps = 120.0

action_item = ['chisel', 'gripper']
obs_item = ['battery']

def trim_lists_in_dicts(dicts):
    # Step 1: Determine the minimum number of lists across all dictionaries
    min_len = min(len(v) for d in dicts for v in d.values())

    # Step 2: Trim the lists in each dictionary to this minimum number
    trimmed_dicts = []
    for d in dicts:
        trimmed_dict = {}
        for k, v in d.items():
            trimmed_dict[k] = v[:min_len]
        trimmed_dicts.append(trimmed_dict)
    
    return trimmed_dicts




base_path = "no_sync/data_chisel_task/2-cleaned_interpolation_with_offset/offset_interpolated_test_traj/"

# # Load data
# dict_of_df_rigid = {}
# dict_of_df_marker = {}


# for file in os.listdir(base_path):

#     if file.endswith(".csv"):
#         path_name = base_path + file
#         data = cfp.DataParser.from_euler_file(file_path = path_name, target_fps=target_fps, filter=True, window_size=15, polyorder=3)

#         marker_data = data.get_marker_Txyz()
#         data_time = data.get_time().astype(float)
#         data_state_dict = data.get_rigid_TxyzRxyz()

#         # use the time and state data to get the velocity data
#         data_velocity_dict = {}
#         for key in data_state_dict.keys():
#             if key != 'battery':
#                 data_velocity = []
#                 for i in range(0, len(data_time)-1):
#                     data_velocity.append((data_state_dict[key][i + 1] - data_state_dict[key][i]) / (data_time[i + 1] - data_time[i]))
#                 velocity_data = pd.DataFrame(data_velocity, columns = [f'{key}_X', f'{key}_Y', f'{key}_Z', f'{key}_x', f'{key}_y', f'{key}_z'])
#                 filtered_velocity = _df.apply_savgol_filter(velocity_data, window_size = 15, polyorder = 3, time_frame= False)
#                 data_velocity_dict[key] = filtered_velocity.values
#             else:
#                 data_velocity_dict[key] = data_state_dict[key]

#         dicts = [data_velocity_dict, marker_data]
#         trimmed_dicts = trim_lists_in_dicts(dicts)

#         dict_of_df_rigid[file] = trimmed_dicts[0]
#         dict_of_df_marker[file] = trimmed_dicts[1]


# item_name = data.rigid_bodies
# marker_name = data.markers

# if len(dict_of_df_rigid) == len(dict_of_df_marker):

#     rigiddataset, index_rigid = _df.episode_combiner(dict_of_df_rigid, item_name)
#     markerdataset, index_marker = _df.episode_combiner(dict_of_df_marker, marker_name)
    # print("Something")
    # print(index[action_item[0]])S


#### if you don't want battery info then just do obs_item = None abd also do clear all outputs and restart the kernal before that and satrt from the top 
# dataset = dproc.TaskStateDataset(Rigiddataset= rigiddataset, Velocitydataset= None, Markerdataset= markerdataset, index=index[action_item[0]], 
#                                  action_item = action_item, obs_item = obs_item,
#                                  marker_item= marker_name,
#                                  pred_horizon=pred_horizon,
#                                  obs_horizon=obs_horizon,
#                                  action_horizon=action_horizon)


# Load data
dict_of_df_rigid = {}
dict_of_df_rigid_velocity = {}
dict_of_df_marker = {}


for file in os.listdir(base_path):

    if file.endswith(".csv"):
        path_name = base_path + file
        data = cfp.DataParser.from_euler_file(file_path = path_name, target_fps=target_fps, filter=True, window_size=15, polyorder=3)

        marker_data = data.get_marker_Txyz()
        data_time = data.get_time().astype(float)
        data_state_dict = data.get_rigid_TxyzRxyz()

        # use the time and state data to get the velocity data
        data_velocity_state_dict = {}
        data_velocity_dict = {}
        for key in data_state_dict.keys():
            if key != 'battery':
                data_velocity = []
                data_velocity_state = []
                for i in range(0, len(data_time) -1):
                    veloctiy_val = (data_state_dict[key][i + 1] - data_state_dict[key][i]) / (data_time[i + 1] - data_time[i])
                    data_velocity.append(veloctiy_val)
                    data_velocity_state.append(np.concatenate((data_state_dict[key][i], veloctiy_val), axis=0).tolist())
                velocity_state_data = pd.DataFrame(data_velocity_state, columns= [f'{key}_X', f'{key}_Y', f'{key}_Z', f'{key}_x', f'{key}_y', f'{key}_z', f'{key}_Xv', f'{key}_Yv', f'{key}_Zv', f'{key}_xv', f'{key}_yv', f'{key}_zv'])
                filtered_velocity_state = _df.apply_savgol_filter(velocity_state_data, window_size = 15, polyorder = 3, time_frame= False)
                data_velocity_state_dict[key] = filtered_velocity_state.values
                velocity_data = pd.DataFrame(data_velocity, columns= [f'{key}_Xv', f'{key}_Yv', f'{key}_Zv', f'{key}_xv', f'{key}_yv', f'{key}_zv'])
                filtered_velocity = _df.apply_savgol_filter(velocity_data, window_size = 15, polyorder = 3, time_frame= False)
                data_velocity_dict[key] = filtered_velocity.values
            else:
                data_velocity_state_dict[key] = data_state_dict[key]


        dicts = [data_velocity_state_dict, data_velocity_dict, marker_data]
        trimmed_dicts = trim_lists_in_dicts(dicts)

        
        dict_of_df_rigid[file] = trimmed_dicts[0]
        dict_of_df_rigid_velocity[file] = trimmed_dicts[1]
        dict_of_df_marker[file] = trimmed_dicts[2]



item_name = data.rigid_bodies
marker_name = data.markers

if len(dict_of_df_rigid) == len(dict_of_df_marker) == len(dict_of_df_rigid_velocity):

    rigiddataset, index_rigid = _df.episode_combiner(dict_of_df_rigid, item_name)
    velocitydataset, index_vel = _df.episode_combiner(dict_of_df_rigid_velocity, action_item)
    markerdataset, index_marker = _df.episode_combiner(dict_of_df_marker, marker_name)
    # print(index[action_item[0]])
    print("Something")
    # print(index[action_item[0]])

# dataset = dproc.TaskStateDataset(Rigiddataset=rigiddataset, Velocitydataset = velocitydataset, Markerdataset= markerdataset, index= index[action_item[0]], 
#                                  action_item = action_item, obs_item = obs_item,
#                                  marker_item= marker_name,
#                                  pred_horizon=pred_horizon,
#                                  obs_horizon=obs_horizon,
#                                  action_horizon=action_horizon)

# # create dataloader
# dataloader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=256,
#     num_workers=1,
#     shuffle=True,
#     # accelerate cpu-gpu transfer
#     pin_memory=True,
#     # don't kill worker process afte each epoch
#     persistent_workers=True
# )

# batch = next(iter(dataloader))
# print("batch['obs'].shape:", batch['obs'].shape)
# print("batch['action'].shape", batch['action'].shape)
