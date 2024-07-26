#@markdown ### **Dataset**
#@markdown
#@markdown Defines `TaskStateDataset` and helper functions
#@markdown
#@markdown The dataset class
#@markdown - Load data (obs, action) from a zarr storage
#@markdown - Normalizes each dimension of obs and action to [-1,1]
#@markdown - Returns
#@markdown  - All possible segments with length `pred_horizon`
#@markdown  - Pads the beginning and the end of each episode with repetition
#@markdown  - key `obs`: shape (obs_horizon, obs_dim)
#@markdown  - key `action`: shape (pred_horizon, action_dim)

import numpy as np
import torch
import pandas as pd
import re
import random
from typing import Union


def generate_sequential_random_sequence(max_value):
    random_sequence = []
    current_value = 0
    
    while current_value < max_value:
        # Generate a random increment (not exceeding 1000 to ensure it doesn't jump too far)
        increment = random.randint(1, min(200, max_value - current_value))
        current_value += increment
        random_sequence.append(current_value)
    
    # If the last value is more than max_value, remove it
    if random_sequence[-1] > max_value:
        random_sequence.pop()
    
    return random_sequence

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = []
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        # print("Start min",min_start)
        max_start = episode_length - sequence_length + pad_after
        # print("Start max",max_start)

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


class TaskStateDataset(torch.utils.data.Dataset):
    def __init__(self, Rigiddataset: Union[list, np.array, None], Velocitydataset: Union[list, np.array, None],
                  Markerdataset: Union[list, np.array, None], index,
                  action_item: Union[list, None], obs_item: Union[list, None], marker_item: Union[list, None],
                 pred_horizon, obs_horizon, action_horizon):

        action = []
        obs = []

        # Ensure action_item, obs_item, and marker_item are not None
        action_item = action_item if action_item is not None else []
        obs_item = obs_item if obs_item is not None else []
        marker_item = marker_item if marker_item is not None else []

        for i in range(index[-1]):
            if Velocitydataset is None:
                a = np.concatenate([Rigiddataset[item][i] for item in action_item]) if Rigiddataset is not None else np.array([])

            else :
                a = np.concatenate([Velocitydataset[item][i] for item in action_item]) if Velocitydataset is not None else np.array([])
            
            b = np.concatenate(
                ([Rigiddataset[item][i] for item in action_item] if Rigiddataset is not None else []) +
                ([Rigiddataset[item][i] for item in obs_item] if Rigiddataset is not None else []) +
                ([Markerdataset[item][i] for item in marker_item] if Markerdataset is not None else [])
            )
            # print(b)
            action.append(a)
            obs.append(b)

    # All demonstration episodes are concatinated in the first dimension N
        action = np.array(action, dtype=np.float64)
        obs = np.array(obs, dtype=np.float64)
        train_data = {
            # (N, action_dim)
            'action': action[:],
            # (N, obs_dim)
            'obs': obs[:]
        }
        # Marks one-past the last index for each episode
        episode_ends = index

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['obs'] = nsample['obs'][:self.obs_horizon,:]
        return nsample
