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
# from robodk import robolink    # RoboDK API
from robomath import *   # Robot toolbox

def distance(current, next):
    distance = math.sqrt((next[0] - current[0])**2 + (next[1] - current[1])**2 + (next[2] - current[2])**2)
    return distance

def sampler(data, sample_size):
    sam = []
    for i in range(0, len(data)):
        if i% sample_size == 0:
            sam.append(data[i])
    # print(sam)
    return sam

def quaternion_2_euler(_quat):
    _pose = quaternion_2_pose(_quat)
    _eul = pose_2_xyzrpw(_pose)
    _eul = np.array(_eul[3:])

    _eul = _eul*pi/180 #convert to radians
    # _eul = normalize_angle(_eul) #normalize the angles from -pi to pi # TODO: comment out
    return _eul

def smooth_angles_chatgpt(angles):
    angles = np.unwrap(angles)
    return angles

def euler_change(Gwxyz) : #receives a list of quaternions
    quaternions = np.array(Gwxyz) #reshape the quaternions to a 2D array
    eulers = []

    for _row in range(quaternions.shape[0]):
        # Create a Rotation object from the quaternion
        _quat = quaternions[_row]
        _eul = quaternion_2_euler(_quat) #normalize the angles from -pi to pi # TODO: comment out
        eulers.append(_eul)
    
    eulers = np.array(eulers)
    smooth_eulers = np.apply_along_axis(smooth_angles_chatgpt, axis=0, arr=eulers)    
    return smooth_eulers.tolist()

def apply_savgol_filter(df: pd.DataFrame, window_size: int, polyorder: int) -> pd.DataFrame:

    smoothed_df = df.copy()
    for col in df.columns:
        smoothed_df[col] = savgol_filter(df[col], window_size, polyorder)
    return smoothed_df


def chisel_data_cleaner(datset_path, base_path, sample):

    for file in os.listdir(datset_path):
        dict_of_lists = {}
        X , Y = [], []

        file_path = os.path.join(datset_path, file)
        data = pd.read_csv(file_path , header = 1)
        # Define the new file name
        new_file_name = re.sub(r'\.csv', '_cleaned.csv', os.path.basename(file_path))

        # Create the new file path with the updated folder
        file_new = os.path.join(base_path, new_file_name)
        data = data.drop(index =1)
        data = data.reset_index(drop=True)

        row1 = data.iloc[0]
        row2 = data.iloc[1]
        row3 = data.iloc[2]
        colums_val = []

        for index, (val , val1) in enumerate(zip(row1, row2)):
            if str(val).startswith('Unlabeled'):
                colums_val.append(index)
            if str(val).startswith('RigidBody') and 'Marker'  not in str(val) and 'Rotation' not in str(val1):
                colums_val.append(index)

        unlabled_data = data.iloc[:,colums_val]

        for idx in range(0, len(unlabled_data.columns), 3):
            col_name = unlabled_data.iloc[0, idx]
            if col_name.startswith('RigidBody'):
                x = float(unlabled_data.iloc[150, idx])
                y = float(unlabled_data.iloc[150, idx + 1])
                z = float(unlabled_data.iloc[150, idx + 2])
                battery_coo = [x, y, z]

            if col_name.startswith('Unlabeled'):
                x = float(unlabled_data.iloc[150, idx])
                y = float(unlabled_data.iloc[150, idx + 1])
                z = float(unlabled_data.iloc[150, idx + 2])
                point = [x, y, z]
                dict_of_lists[col_name] = point

        filtered_dict = {key: [value for value in values if np.isfinite(value)] for key, values in dict_of_lists.items() if any(np.isfinite(value) for value in values)}

        P = np.array(battery_coo)
        Points = np.array(filtered_dict)

        vec = {}
        for key in filtered_dict.keys():
            vector_ab = P - filtered_dict[key]
            vec[key] = np.round(vector_ab[0], 2), np.round(vector_ab[2], 2)

        common_values = {
            'A1': (-0.16, -0.08),
            'A2': (-0.16, 0.0),
            'A3': (-0.16, 0.08),
            'B1': (-0.0, -0.08),
            'B2': (-0.0, 0.0),
            'B3': (-0.0, 0.08),
            'C1': (0.15, -0.08),
            'C2': (0.15, 0.0),
            'C3': (0.15, 0.07)}

        matching_keys = {}

        for key, value in vec.items():
            for common_key, common_value in common_values.items():
                if value == common_value:
                    matching_keys[common_key] = key
                    break

        combined_values = []
        columns_to_drop = [] 

        for index, (a, b, c) in enumerate(zip(row1, row2, row3)):
            # print(str(b) + '_' + str(c))
            if str(b) + '_' + str(c) in ('Rotation_X', 'Rotation_Y', 'Rotation_Z', 'Rotation_W'):
                if 'Marker' in str(a):
                    columns_to_drop.append(index)
                elif 'RigidBody' in str(a):
                    combined_values.append('battery_' + c.lower())
                else:
                    combined_values.append(str(a).split("_")[0] + '_' + c.lower())
            else:
                if str(a) in matching_keys.values():
                    combined_values.append(next((key for key, value in matching_keys.items() if value == str(a)), None) + '_' + c)
                elif 'Marker' in str(a) or 'Active' in str(a):
                    columns_to_drop.append(index)
                elif 'RigidBody' in str(a):
                    combined_values.append('battery_' + c)
                elif str(a) not in matching_keys.values() and str(a).startswith('Unlabeled'):
                    columns_to_drop.append(index)
                else:
                    combined_values.append(str(a).split("_")[0] + '_' + c)
                    
        columns_to_drop.append(0)
        columns_to_drop.append(1)
        data = data.drop(data.columns[columns_to_drop], axis=1)
        data.columns = combined_values[2:]
        data = data.drop(index =0)
        data = data.drop(index =1)
        data = data.drop(index =2)
        data = data.dropna()
        data = data.reset_index(drop=True)
        smoothed_df = {}
        print(new_file_name)
        for col in data.columns:
            asd = sampler(data[col], sample)
            smoothed_df[col] = savgol_filter(asd, 15, 3)

        smoothed_data = pd.DataFrame(smoothed_df)
        smoothed_data.to_csv(f'{file_new}', index=False)


def extract_data_chisel(data_path):

    CXYZ, Cwxyz, GXYZ, Gwxyz, BXYZ, Bwxyz, A1XYZ, A2XYZ, A3XYZ, B1XYZ, B2XYZ, B3XYZ, C1XYZ, C2XYZ, C3XYZ = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    index = []
    desired_order = ['chisel_x', 'chisel_y', 'chisel_z', 'chisel_w', 'chisel_X', 'chisel_Y',
            'chisel_Z', 'gripper_x', 'gripper_y', 'gripper_z', 'gripper_w', 'gripper_X',
            'gripper_Y', 'gripper_Z', 'battery_x', 'battery_y', 'battery_z', 'battery_w',
            'battery_X', 'battery_Y', 'battery_Z', 'A1_X', 'A1_Y', 'A1_Z', 'A2_X', 'A2_Y',
            'A2_Z', 'A3_X', 'A3_Y', 'A3_Z', 'B1_X', 'B1_Y', 'B1_Z', 'B2_X', 'B2_Y', 'B2_Z',
            'B3_X', 'B3_Y', 'B3_Z', 'C1_X', 'C1_Y', 'C1_Z', 'C2_X', 'C2_Y', 'C2_Z',
            'C3_X', 'C3_Y', 'C3_Z']


    for file in os.listdir(data_path):
        # print(file)

        file_path = os.path.join(data_path, file)
        data = pd.read_csv(file_path)
        dict_of_lists = data.to_dict('list')
        chisel_pos = [tuple(item) for item in zip(dict_of_lists[desired_order[0]], dict_of_lists[desired_order[1]], dict_of_lists[desired_order[2]])]
        # chisel_pos = sampler(chisel_pos, sample_size)
        chisel_rot = [tuple(item) for item in zip(dict_of_lists[desired_order[3]], dict_of_lists[desired_order[4]], dict_of_lists[desired_order[5]], dict_of_lists[desired_order[6]])]
        # chisel_rot = sampler(chisel_rot, sample_size)
        gripper_pos = [tuple(item) for item in zip(dict_of_lists[desired_order[7]], dict_of_lists[desired_order[8]], dict_of_lists[desired_order[9]])]
        # gripper_pos = sampler(gripper_pos, sample_size)
        gripper_rot = [tuple(item) for item in zip(dict_of_lists[desired_order[10]], dict_of_lists[desired_order[11]], dict_of_lists[desired_order[12]], dict_of_lists[desired_order[13]])]
        # gripper_rot = sampler(gripper_rot, sample_size)
        battery_pos = [tuple(item) for item in zip(dict_of_lists[desired_order[14]], dict_of_lists[desired_order[15]], dict_of_lists[desired_order[16]])]
        # battery_pos = sampler(battery_pos, sample_size)
        battery_rot = [tuple(item) for item in zip(dict_of_lists[desired_order[17]], dict_of_lists[desired_order[18]], dict_of_lists[desired_order[19]], dict_of_lists[desired_order[20]])]
        # battery_rot = sampler(battery_rot, sample_size)
        A1_pos = [tuple(item) for item in zip(dict_of_lists[desired_order[21]], dict_of_lists[desired_order[22]], dict_of_lists[desired_order[23]])]
        # A1_pos = sampler(A1_pos, sample_size)
        A2_pos = [tuple(item) for item in zip(dict_of_lists[desired_order[24]], dict_of_lists[desired_order[25]], dict_of_lists[desired_order[26]])]
        # A2_pos = sampler(A2_pos, sample_size)
        A3_pos = [tuple(item) for item in zip(dict_of_lists[desired_order[27]], dict_of_lists[desired_order[28]], dict_of_lists[desired_order[29]])]
        # A3_pos = sampler(A3_pos, sample_size)
        B1_pos = [tuple(item) for item in zip(dict_of_lists[desired_order[30]], dict_of_lists[desired_order[31]], dict_of_lists[desired_order[32]])]
        # B1_pos = sampler(B1_pos, sample_size)
        B2_pos = [tuple(item) for item in zip(dict_of_lists[desired_order[33]], dict_of_lists[desired_order[34]], dict_of_lists[desired_order[35]])]
        # B2_pos = sampler(B2_pos, sample_size)
        B3_pos = [tuple(item) for item in zip(dict_of_lists[desired_order[36]], dict_of_lists[desired_order[37]], dict_of_lists[desired_order[38]])]
        # B3_pos = sampler(B3_pos, sample_size)
        C1_pos = [tuple(item) for item in zip(dict_of_lists[desired_order[39]], dict_of_lists[desired_order[40]], dict_of_lists[desired_order[41]])]
        # C1_pos = sampler(C1_pos, sample_size)
        C2_pos = [tuple(item) for item in zip(dict_of_lists[desired_order[42]], dict_of_lists[desired_order[43]], dict_of_lists[desired_order[44]])]
        # C2_pos = sampler(C2_pos, sample_size)
        C3_pos = [tuple(item) for item in zip(dict_of_lists[desired_order[45]], dict_of_lists[desired_order[46]], dict_of_lists[desired_order[47]])]
        # C3_pos = sampler(C3_pos, sample_size)

        # Now to modify each sublist in chisel_pos
        chisel_pos = [(sublist[2], sublist[0], sublist[1]) for sublist in chisel_pos]
        gripper_pos = [(sublist[2], sublist[0], sublist[1]) for sublist in gripper_pos]
        battery_pos = [(sublist[2], sublist[0], sublist[1]) for sublist in battery_pos]
        A1_pos = [(sublist[2], sublist[0], sublist[1]) for sublist in A1_pos]
        A2_pos = [(sublist[2], sublist[0], sublist[1]) for sublist in A2_pos]
        A3_pos = [(sublist[2], sublist[0], sublist[1]) for sublist in A3_pos]
        B1_pos = [(sublist[2], sublist[0], sublist[1]) for sublist in B1_pos]
        B2_pos = [(sublist[2], sublist[0], sublist[1]) for sublist in B2_pos]
        B3_pos = [(sublist[2], sublist[0], sublist[1]) for sublist in B3_pos]
        C1_pos = [(sublist[2], sublist[0], sublist[1]) for sublist in C1_pos]
        C2_pos = [(sublist[2], sublist[0], sublist[1]) for sublist in C2_pos]
        C3_pos = [(sublist[2], sublist[0], sublist[1]) for sublist in C3_pos]

        # For rotation, if you have w, x, y, z and want w, x, y, z re-ordered to w, x, y, z:
        chisel_rot = [(sublist[3], sublist[2], sublist[0], sublist[1]) for sublist in chisel_rot]
        gripper_rot = [(sublist[3], sublist[2], sublist[0], sublist[1]) for sublist in gripper_rot]
        battery_rot = [(sublist[3], sublist[2], sublist[0], sublist[1]) for sublist in battery_rot]

        # Assuming euler_change() is applied correctly elsewhere
        chisel_rot = euler_change(chisel_rot)
        gripper_rot = euler_change(gripper_rot)
        battery_rot = euler_change(battery_rot)


        CXYZ.extend(chisel_pos)
        print(file)
        index.append(len(CXYZ))
        print(len(CXYZ))
        Cwxyz.extend(chisel_rot)
        GXYZ.extend(gripper_pos)
        Gwxyz.extend(gripper_rot)
        BXYZ.extend(battery_pos)
        Bwxyz.extend(battery_rot)
        A1XYZ.extend(A1_pos)
        A2XYZ.extend(A2_pos)
        A3XYZ.extend(A3_pos)
        B1XYZ.extend(B1_pos)
        B2XYZ.extend(B2_pos)
        B3XYZ.extend(B3_pos)
        C1XYZ.extend(C1_pos)
        C2XYZ.extend(C2_pos)
        C3XYZ.extend(C3_pos)

    return CXYZ, Cwxyz, GXYZ, Gwxyz, BXYZ, Bwxyz, A1XYZ, A2XYZ, A3XYZ, B1XYZ, B2XYZ, B3XYZ, C1XYZ, C2XYZ, C3XYZ, index
    
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
class RealStateDataset(torch.utils.data.Dataset):
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



class ChiselStateDataset(torch.utils.data.Dataset):
    def __init__(self, dataset,  base_path,
                 pred_horizon, obs_horizon, action_horizon, sample_size):

        # read from zarr dataset
        chisel_data_cleaner(dataset, base_path, sample_size)

        # for key in result_dict:
        CXYZ, Cwxyz, GXYZ, Gwxyz, BXYZ, Bwxyz, A1XYZ, A2XYZ, A3XYZ, B1XYZ, B2XYZ, B3XYZ, C1XYZ, C2XYZ, C3XYZ, index = extract_data_chisel(base_path)

        action = []
        obs = []
        for i in range(len(CXYZ)):
            # a = []
            a = [*Cwxyz[i], *CXYZ[i], *Gwxyz[i], *GXYZ[i]]
            print(a)
            b = [*Cwxyz[i], *CXYZ[i], *Gwxyz[i], *GXYZ[i], *Bwxyz[i], *BXYZ[i], *A1XYZ[i], *A2XYZ[i], *A3XYZ[i], *B1XYZ[i], *B2XYZ[i], *B3XYZ[i], *C1XYZ[i], *C2XYZ[i], *C3XYZ[i]]
            
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
