#@markdown ### **Dataset**
#@markdown
#@markdown Defines `PushTStateDataset` and helper functions
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
    
def extract_data_handover(result_dict, sample_size):

    GXYZ, Gwxyz, SXYZ, Swxyz, BXYZ, Bwxyz, DXYZ, Dwxyz  = [], [], [], [], [], [], [], []
    indexes = []
    diff = []
    mins = []
    value_to_indexes = {}
    indexes_in_list2 = []

    for key in result_dict:
        path_value = result_dict[key]['Path']
        start = int(result_dict[key]['start_frame'])
        end = int(result_dict[key]['end_frame'])

        data  = pd.read_csv(path_value)
        data = data.reset_index(drop=False)
        data = data.drop(index =0)
        data = data.drop(index =2)
        data = data.reset_index(drop=True)

        row1 = data.iloc[0]
        row2 = data.iloc[1]
        row3 = data.iloc[2]

        combined_values = []
        for a, b, c in zip(row1, row2, row3):
            combined_values.append(str(a) + '_' + str(b) + '_' + str(c))

        data.columns = combined_values
        data = data.drop(index =0)
        data = data.drop(index =1)
        data = data.drop(index =2)
        data = data.drop(data.columns[:2], axis=1)
        # print(result_dict)
        data = data.iloc[start:end]
        data = data.dropna()
        data = data.reset_index(drop=True)

        # Regular expression pattern to match columns starting with 'gripper_1_Rotation'
        pattern1 = re.compile(r'GRIPPER_2_Rotation')
        pattern2 = re.compile(r'GRIPPER_2_Position')
        pattern3 = re.compile(r'diff_scooper_2_2_Rotation')
        pattern4 = re.compile(r'diff_scooper_2_2_Position')
        pattern5 = re.compile(r'box3_Rotation')
        pattern6 = re.compile(r'box3_Position')
        pattern7 = re.compile(r'bucket_SC_Rotation')
        pattern8 = re.compile(r'bucket_SC_Position')

        # Filter columns using regex pattern and extract values into a list
        a = data.filter(regex=pattern1).values.astype('float64').tolist()
        print(a)
        a = sampler(a, sample_size)
        b = data.filter(regex=pattern2).values.astype('float64').tolist()
        b = sampler(b, sample_size)
        c = data.filter(regex=pattern3).values.astype('float64').tolist()
        c = sampler(c, sample_size)
        d = data.filter(regex=pattern4).values.astype('float64').tolist()
        d = sampler(d, sample_size)
        e = data.filter(regex=pattern5).values.astype('float64').tolist()
        e = sampler(e, sample_size)
        f = data.filter(regex=pattern6).values.astype('float64').tolist()
        f = sampler(f, sample_size)
        g = data.filter(regex=pattern7).values.astype('float64').tolist()
        g = sampler(g, sample_size)
        h = data.filter(regex=pattern8).values.astype('float64').tolist()
        h = sampler(h, sample_size)

        for sublist in b:
            y = sublist[0] 
            z= sublist[1]
            x = sublist[2]
            sublist[0] = x
            sublist[1] = y
            sublist[2] = z    

        for sublist in d:
            y = sublist[0]
            z= sublist[1]
            x = sublist[2]
            sublist[0] = x
            sublist[1] = y
            sublist[2] = z  

        for sublist in f:
            y = sublist[0]
            z= sublist[1]
            x = sublist[2]
            sublist[0] = x
            sublist[1] = y
            sublist[2] = z  

        for sublist in h:
            y = sublist[0]
            z= sublist[1]
            x = sublist[2]
            sublist[0] = x
            sublist[1] = y
            sublist[2] = z  

        for sublist in a:
            y = sublist[0]
            z = sublist[1]
            x = sublist[2]
            w = sublist[3]
            sublist[0] = w
            sublist[1] = x
            sublist[2] = y    
            sublist[3] = z    

        a = euler_change(a)
        # print(a)

        for sublist in c:
            y = sublist[0]
            z = sublist[1]
            x = sublist[2]
            w = sublist[3]
            sublist[0] = w
            sublist[1] = x
            sublist[2] = y    
            sublist[3] = z   

        c = euler_change(c)

        for sublist in e:
            y = sublist[0]
            z = sublist[1]
            x = sublist[2]
            w = sublist[3]
            sublist[0] = w
            sublist[1] = x
            sublist[2] = y    
            sublist[3] = z  

        e = euler_change(e)

        for sublist in g:
            y = sublist[0]
            z = sublist[1]
            x = sublist[2]
            w = sublist[3]
            sublist[0] = w
            sublist[1] = x
            sublist[2] = y    
            sublist[3] = z

        g = euler_change(g)

        GXYZ.extend(b)
        indexes.append(len(GXYZ))
        Gwxyz.extend(a)
        SXYZ.extend(d)
        Swxyz.extend(c)
        BXYZ.extend(f)
        Bwxyz.extend(e)
        DXYZ.extend(h)
        Dwxyz.extend(g)


    for i in range(min(len(GXYZ), len(BXYZ))):
        diff_s = distance(GXYZ[i], BXYZ[i])
        diff.append(diff_s)

    GA = np.full_like(diff, -1)
    min_values = sorted(set(diff))

    for index, value in enumerate(min_values):
        if value < 0.01:
            mins.append(value)
            
    # Populate the dictionary with list2 values and their indexes
    for index, value in enumerate(diff):
        if value in value_to_indexes:
            value_to_indexes[value].append(index)
        else:
            value_to_indexes[value] = [index]

    # Find indexes in list2 corresponding to values in list1
    for value in mins:
        if value in value_to_indexes:
            indexes_in_list2.extend(value_to_indexes[value])

    for i in range (min(indexes_in_list2), max(indexes_in_list2)+1):
        GA[i] = 1

    # return  b, a, d, c, f, e, h, g
    return GXYZ, Gwxyz, SXYZ, Swxyz, BXYZ, Bwxyz, DXYZ, Dwxyz, indexes, GA.tolist()

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


def get_data_stats_handover(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    
    # print(len(stats['min']))
    # print(len(stats['max']))

    if len(stats['min']) == 13:
        indices_to_change_action = [0, 1, 2, 6, 7, 8]
        stats['min'][indices_to_change_action] =  -np.pi
        stats['max'][indices_to_change_action] =  np.pi
    elif len(stats['min']) == 25:
        indices_to_change_obs = [0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20]
        stats['min'][indices_to_change_obs] =  -np.pi
        stats['max'][indices_to_change_obs] =  np.pi
        
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

# dataset
class HandOverStateDataset(torch.utils.data.Dataset):
    def __init__(self, dataset,  base_path,
                 pred_horizon, obs_horizon, action_horizon, sample_size):
        
        # read from zarr dataset
        list = dataset
        # Base path
        collums = list.columns

        result_dict = {}
        count = 0
        for i in range(len(list)):
            if list[collums[3]][i] == 'accept':
                result_dict[count] = {
                    'Path': base_path + str(list[collums[0]][i]) + '.csv',
                    'start_frame': list[collums[1]][i],
                    'end_frame': list[collums[2]][i],
                    'Note': list[collums[4]][i]
                }
                count += 1

        # for key in result_dict:
        GXYZ, Gwxyz, SXYZ, Swxyz, BXYZ, Bwxyz, DXYZ, Dwxyz, index , GA = extract_data_handover(result_dict, sample_size)

        action = []
        obs = []
        for i in range(len(GXYZ)):
            # a = []
            a = Gwxyz[i] + GXYZ[i] + Swxyz[i] + SXYZ[i]
            a.append(GA[i])
            b = Gwxyz[i] + GXYZ[i] + Swxyz[i] + SXYZ[i] + Dwxyz[i] + DXYZ[i] + Bwxyz[i] + BXYZ[i]
            b.append(GA[i])
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
        # episode_ends = generate_sequential_random_sequence(3585)
        episode_len = index

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_len,
            sequence_length=pred_horizon,
            # add padding such that each timestep in the dataset are seen
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)
        
        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats_handover(data)
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
        # print("gett item")
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
    
# dataset
class PushTStateDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path,
                 pred_horizon, obs_horizon, action_horizon):

        # read from zarr dataset
        dataset_root = zarr.open(dataset_path, 'r')

        # All demonstration episodes are concatinated in the first dimension N
        train_data = {
            # (N, action_dim)
            'action': dataset_root['data']['action'][:],
            # (N, obs_dim)
            'obs': dataset_root['data']['state'][:]
        }
        # Marks one-past the last index for each episode
        episode_ends = dataset_root['meta']['episode_ends'][:]

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


class TaskStateDataset(torch.utils.data.Dataset):
    def __init__(self, Tooldataset: Union[list, np.array, None], Rigiddataset: Union[list, np.array, None],
                  Markerdataset: Union[list, np.array, None], index,
                 pred_horizon, obs_horizon, action_horizon):

        action = []
        obs = []
        for i in range(index[-1]):
            # a = []
            a = [*Tooldataset[i]]
            print(a)
            b = [*Tooldataset[i], *Rigiddataset[i], *Markerdataset[i]]
            
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
