# diffusion policy import
from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import pandas as pd
import math
import os
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



base_path = "/home/cam/Documents/diffusion_policy_cam/diffusion_pipline/data_chisel_task/cleaned_traj/"

# Load data
dict_of_df_rigid = {}
dict_of_df_marker = {}


for file in os.listdir(base_path):
    if file.endswith(".csv") and file.startswith("cap"):
        path_name = base_path + file
        data = cfp.DataParser.from_quat_file(file_path = path_name, target_fps=target_fps, filter=False, window_size=15, polyorder=3)
        dict_of_df_rigid[file] = data.get_rigid_TxyzRxyz()
        dict_of_df_marker[file] = data.get_marker_Txyz()
        
item_name = data.rigid_bodies
marker_name = data.markers

if len(dict_of_df_rigid) == len(dict_of_df_marker):

    rigiddataset, index = _df.episode_combiner(dict_of_df_rigid, item_name)
    markerdataset, _ = _df.episode_combiner(dict_of_df_marker, marker_name)
    # print(index[action_item[0]])

dataset = dproc.TaskStateDataset(rigiddataset, markerdataset, index[action_item[0]], 
                                 action_item = action_item, obs_item = obs_item,
                                 marker_item= marker_name,
                                 pred_horizon=pred_horizon,
                                 obs_horizon=obs_horizon,
                                 action_horizon=action_horizon)

# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    num_workers=1,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)

batch = next(iter(dataloader))
print("batch['obs'].shape:", batch['obs'].shape)
print("batch['action'].shape", batch['action'].shape)
