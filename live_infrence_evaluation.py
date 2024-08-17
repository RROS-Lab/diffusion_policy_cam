# diffusion policy import
from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import pandas as pd
import csv
import torch
import torch.nn as nn
import collections
import rclpy
import submodules.data_filter as _df
import diffusion_pipline.data_processing as dproc
import diffusion_pipline.model as md
import submodules.cleaned_file_parser as cfp
from rclpy.node import Node
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from std_msgs.msg import Int32MultiArray




class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


#@markdown ### **Network Demo**
def predicted_action(obs_deque, action_item, statistics, obs_horizon,
                        pred_horizon, action_horizon, action_dim, 
                        noise_scheduler, num_diffusion_iters,
                        ema_noise_pred_net, device):   
    
    stats = statistics


    # save visualization and rewards
    B = 1
    # stack the last obs_horizon (2) number of observations
    obs_seq = np.stack(obs_deque)
    # normalize observation
    nobs = dproc.normalize_data(obs_seq, stats=stats['obs'])
    # device transfer
    nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)
    # infer action
    with torch.no_grad():
        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)
        
        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, pred_horizon, action_dim), device=device)
        naction = noisy_action

        # init scheduler
        noise_scheduler.set_timesteps(num_diffusion_iters)

        for k in noise_scheduler.timesteps:
            # predict noise
            noise_pred = ema_noise_pred_net(
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

    # unnormalize action
    naction = naction.detach().to('cpu').numpy()
    naction = naction[0]
    action_pred = dproc.unnormalize_data(naction, stats=stats['action'])

    # only take action_horizon number of actions
    start = obs_horizon - 1
    end = start + action_horizon
    action = action_pred[start:end,:]
    grouped_actions = {name: [] for name in action_item}

    # Divide the 18 elements in each row into three groups
    for i in range(len(action)):
        for j in range(len(action_item)):
            grouped_actions[action_item[j]].append(action[i, 6*j:6*(j+1)])

    return grouped_actions


def _pred_traj(first_obs, com_obs_part, action_item, statistics, obs_horizon,
                        pred_horizon, action_horizon, action_dim, 
                        noise_scheduler, num_diffusion_iters,
                        ema_noise_pred_net, device):
    
    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [first_obs] * obs_horizon, maxlen=obs_horizon)
    
    action =  predicted_action(obs_deque, action_item,  statistics, obs_horizon,
                                pred_horizon, action_horizon, action_dim, noise_scheduler, num_diffusion_iters,
                                ema_noise_pred_net, device)
    
    battery_gripper = action['battery'] + action['gripper']
    minimal_client = MinimalClientAsync()
    response = minimal_client.send_request(battery_gripper)
    
    traj = []
    for i in range(len(action)):
        action_last = list(action[i])
        obs_deque.append(action_last + com_obs_part)
        traj.append(action_last + com_obs_part)
        
        
    return traj
    


def _create_csv_file(trajectories, save_dir, action_item, marker_name, save_type):
    
    for index, key in enumerate(trajectories.keys()):
        print(index)
        # Define the file path or name
        file_path = f'{save_dir}/pred_{key}'

        # add first rows
        _params = {
            'QUAT': {'len':7,
                        'dof': ['X', 'Y', 'Z', 'w', 'x', 'y', 'z']},
            'EULER': {'len':6,
                        'dof': ['X', 'Y', 'Z', 'x', 'y', 'z']},
            'Vel': {'len':6,
                        'dof': ['Xv', 'Yv', 'Zv', 'xv', 'yv', 'zv']}
        }
        
        _SUP_HEADER_ROW = (["RigidBody"] * len(action_item) * _params[save_type]['len'] + ["Marker"] * len(marker_name) * 3)
        _FPS_ROW = ["FPS", target_fps] + [0.0]*(len(_SUP_HEADER_ROW) - 2)
        _rb_col_names = [f"{rb}_{axis}" for rb in action_item for axis in _params[save_type]['dof']]
        # _obs_col_name = [f"{rb}_{axis}" for rb in obs_item for axis in _params[save_type]['dof']]
        _mk_col_names = [f"{mk}_{axis}" for mk in marker_name for axis in ['X', 'Y', 'Z']]
        _HEADER_ROW = _rb_col_names + _mk_col_names
            # Open the file in write mode
        with open(file_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(_SUP_HEADER_ROW)
            writer.writerow(_FPS_ROW)
            writer.writerow(_HEADER_ROW)
            writer.writerows(trajectories[key])


def _model_initalization(action_dim, obs_dim, obs_horizon, pred_horizon, num_epochs, len_dataloader, num_diffusion_iters) -> nn.Module:
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
    # num_diffusion_iters = 100
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
        num_training_steps=len_dataloader * num_epochs
    )

    ema_noise_pred_net = noise_pred_net

    return ema_noise_pred_net, noise_scheduler, device, ema, optimizer, lr_scheduler




if __name__ == "__main__":
    
    checkpoint_path = '/home/cam/Documents/raj/diffusion_policy_cam/no-sync/checkpoints/checkpoint_3Body_9_markers_960_SSSS_epoch_119.pth'

    checkpoint = torch.load(checkpoint_path)

    # Parameters corrsponding to
    num_epochs =checkpoint['num_epochs']
    obs_dim = checkpoint['obs_dim']
    action_dim = checkpoint['action_dim']
    # parameters
    pred_horizon = checkpoint['pred_horizon']
    obs_horizon = checkpoint['obs_horizon']
    action_horizon = checkpoint['action_horizon']
    target_fps = checkpoint['target_fps']

    action_item = checkpoint['action_item']
    obs_item = checkpoint['obs_item']
    marker_name = checkpoint['marker_name']
    start_epoch = checkpoint['epoch'] + 1
    len_dataloader = checkpoint['len_dataloader']   
    num_diffusion_iters = checkpoint['num_diffusion_iters']
    
    noise_pred_net, noise_scheduler, device, ema, optimizer, lr_scheduler = _model_initalization(action_dim, obs_dim, obs_horizon, pred_horizon, num_epochs, len_dataloader, num_diffusion_iters)
    
    noise_pred_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    ema.load_state_dict(checkpoint['ema_state_dict'])
    

    