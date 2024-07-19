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
sample_size = 8

action_item = ['chisel', 'gripper']
obs_item = ['battery']

# create network object
noise_pred_net = md.ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

# example inputs
noised_action = torch.randn((1, pred_horizon, action_dim))
obs = torch.zeros((1, obs_horizon, obs_dim))
diffusion_iter = torch.zeros((1,))

# the noise prediction network
# takes noisy action, diffusion iteration and observation as input
# predicts the noise added to action
noise = noise_pred_net(
    sample=noised_action,
    timestep=diffusion_iter,
    global_cond=obs.flatten(start_dim=1))

# illustration of removing noise
# the actual noise removal is performed by NoiseScheduler
# and is dependent on the diffusion noise schedule
denoised_action = noised_action - noise

# for this demo, we use DDPMScheduler with 100 diffusion iterations
num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

# device transfer
device = torch.device('cuda')
_ = noise_pred_net.to(device)

# create dataset from file
# path_name = "/home/cam/Downloads/Supporting Data - Sheet1.csv"
base_path = "/home/cam/Documents/diffusion_policy_cam/diffusion_pipline/data_chisel_task/cleaned_traj/"

# Load data
dict_of_df_rigid = {}
dict_of_df_marker = {}


for file in os.listdir(base_path):
    if file.endswith(".csv") and file.startswith("cap"):
        path_name = base_path + file
        data = cfp.DataParser.from_quat_file(file_path = path_name, target_fps=120.0, filter=False, window_size=15, polyorder=3)
        dict_of_df_rigid[file] = data.get_rigid_TxyzRxyz()
        dict_of_df_marker[file] = data.get_marker_Txyz()
        
item_name = data.rigid_bodies
marker_name = data.markers

if len(dict_of_df_rigid) == len(dict_of_df_marker):

    rigiddataset, index = _df.episode_combiner(dict_of_df_rigid, item_name)
    markerdataset, _ = _df.episode_combiner(dict_of_df_marker, marker_name)
    print(index[action_item[0]])

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

#@markdown ### **Training**
#@markdown
#@markdown Takes about an hour. If you don't want to wait, skip to the next cell
#@markdown to load pre-trained weights

num_epochs =400
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_interval = 3600
last_checkpoint_time = time.time()
# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
ema = EMAModel(
    parameters=noise_pred_net.parameters(),
    power=0.75)

# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=noise_pred_net.parameters(),
    lr=1e-4, weight_decay=1e-6)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=200,
    num_training_steps=len(dataloader) * num_epochs
)

with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    # epoch loop
    epoch_loss = []
    batch_loss_per_epoch = []

    for epoch_idx in tglobal:
        batch_loss = []
        batch_noise = []
        # batch loop
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:

            for nbatch in tepoch:
                # data normalized in dataset
                # device transfer
                nobs = nbatch['obs']
                naction = nbatch['action']
                B = nobs.shape[0]

                # observation as FiLM conditioning
                # (B, obs_horizon, obs_dim)
                obs_cond = nobs[:,:obs_horizon,:]
                # (B, obs_horizon * obs_dim)
                obs_cond = obs_cond.flatten(start_dim=1).float().to(device)
                # print(obs_cond.type())

                # sample noise to add to actions
                # noise = torch.randn(naction.shape, device=device)
                noise = torch.randn(naction.shape)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,)
                ).long()

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = noise_scheduler.add_noise(
                    naction, noise, timesteps)
                
                noise = noise.to(device)
                
                timesteps = timesteps.to(device)

                # print(noisy_actions.type())
                noisy_actions = noisy_actions.type(torch.FloatTensor).to(device)
                # print(noisy_actions.type())

                # predict the noise residual
                noise_pred = noise_pred_net(
                    noisy_actions, timesteps, global_cond=obs_cond)
                
                batch_noise.append(noise_pred)

                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(noise_pred_net)
                # print(ema.state_dict)

                # logging
                loss_cpu = loss.item()
                batch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)

        # save checkpoint
        # went to the emma model library and added state_dict to the model
        current_time = time.time()
        if current_time - last_checkpoint_time > checkpoint_interval:
        # if epoch_idx == 2:
            # Save model checkpoint
            # checkpoint_path = os.path.join(checkpoint_dir, f'BOX_GRIP_checkpoint_epoch_{epoch_idx}.pth')
            checkpoint_path = os.path.join(checkpoint_dir, f'T_checkpoint_epoch_{epoch_idx}.pth')

            torch.save({
                'epoch': epoch_idx,
                'model_state_dict': noise_pred_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'ema_state_dict': ema.state_dict,
                'loss': loss.cpu().detach().numpy(),
            }, checkpoint_path)
            print(f'Checkpoint saved at epoch {epoch_idx}')
            last_checkpoint_time = current_time
        elif epoch_idx == num_epochs:
            # Save model checkpoint
            # checkpoint_path = os.path.join(checkpoint_dir, f'BOX_GRIP_checkpoint_epoch_{epoch_idx}.pth')
            checkpoint_path = os.path.join(checkpoint_dir, f'T_checkpoint_epoch_{epoch_idx}.pth')
            torch.save({
                'epoch': epoch_idx,
                'model_state_dict': noise_pred_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'ema_state_dict': ema.state_dict,
                'loss': loss.cpu().detach().numpy(),
            }, checkpoint_path)
            print(f'Checkpoint saved at epoch {epoch_idx}')
            last_checkpoint_time = current_time
            
        tglobal.set_postfix(loss=np.mean(batch_loss))
        epoch_loss.append(np.mean(batch_loss))
        batch_loss_per_epoch.append(batch_loss)

# Weights of the EMA model
# is used for inference
ema_noise_pred_net = noise_pred_net
# ema.copy_to(ema_noise_pred_net.parameters())