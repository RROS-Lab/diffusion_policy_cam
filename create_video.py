import math
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
# from scipy.spatial.transform import Rotation
from matplotlib.backends.backend_pdf import PdfPages
from submodules.robomath import *
import cv2
import os

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


def euler_change(Gwxyz):
    quaternion = np.array(Gwxyz)
    euler_angles = []

    for i in range(len(quaternion)):
        # Create a Rotation object from the quaternion
        pos_i = quaternion_2_pose(quaternion[i])
        # print("pos_i")
        # print(pos_i)
        eul_i = pose_2_xyzrpw(pos_i)
        # print("eul_i")
        # print(eul_i)
        # Convert to Euler angles (in radians)
        euler_angles.append(eul_i[3:])  # 'xyz' specifies the order of rotations

    return euler_angles

def extract_data(result_dict):
    sample_size = 8
    path_value = result_dict['Path']
    start = int(result_dict['start_frame'])
    end = int(result_dict['end_frame'])

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

    for sublist in g:
        y = sublist[0]
        z = sublist[1]
        x = sublist[2]
        w = sublist[3]
        sublist[0] = w
        sublist[1] = x
        sublist[2] = y    
        sublist[3] = z   

    # return  b, a, d, c

    return {'GR':a,
    'GP':b,
    'SR':c,
    'SP':d,
    'BxR':e,
    'BxP':f,
    'BuR':g,
    'BuP':h}

def XYZwxyz_2_Pose(XYZwxyz):
    '''
    XYZwxyz --> Pose (Homegeneous 4X4 Matrix)
    this converts XYZ Quat to corresponding Homogeneous Matrix
    '''
    # print(XYZwxyz[3:])
    _pose_matrix_without_XYZ = quaternion_2_pose(XYZwxyz[3:])
    for i in range(3):
        # print(XYZwxyz[i])
        # print(XYZwxyz[i] - 1000)
        if i == 0:
            _pose_matrix_without_XYZ[i,3] = XYZwxyz[i] - 0
        else:
            _pose_matrix_without_XYZ[i,3] = XYZwxyz[i]
    return _pose_matrix_without_XYZ

def Pose_2_XYZwxyz(pose):
    wxyz=pose_2_quaternion(pose)
    XYZ = [pose[i,3] for i in range(3)]
    return [*XYZ, *wxyz]


def normalize_positions(datasets):
    all_x = []
    all_y = []
    for dataset in datasets:
        for position in dataset:
            all_x.append(position[0])
            all_y.append(position[1])
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    normalized_datasets = []
    for dataset in datasets:
        normalized_dataset = []
        for position in dataset:
            normalized_x = (position[0] - min_x) / (max_x - min_x)
            normalized_y = (position[1] - min_y) / (max_y - min_y)
            normalized_dataset.append((normalized_x, normalized_y))
        normalized_datasets.append(normalized_dataset)
    
    return normalized_datasets

def create_frames(datasets, frame_dir='./vids/frames', width=800, height=600, **kwargs):
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)

    marker_list = kwargs.get('marker_list', ['o' for _ in datasets])
    normalized_datasets = normalize_positions(datasets)

    for time_step in range(len(normalized_datasets[0])):
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        for obj_i in range(len(normalized_datasets)):
            x, y = normalized_datasets[obj_i][time_step]
            ax.plot(x, y, marker_list[obj_i])
        
        frame_path = os.path.join(frame_dir, f'frame_{time_step:04d}.png')
        plt.savefig(frame_path)
        plt.close(fig)

def create_video_from_frames(frame_dir='frames', fps=30, video_filename='./vids/output.avi', width=800, height=600):
    frame_files = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.png')])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    for frame_file in frame_files:
        img = cv2.imread(frame_file)
        img = cv2.resize(img, (width, height))
        video.write(img)

    video.release()


############################################################################################################
### Main ###
############################################################################################################

print("..........")
path_name = "./../../datasets/dataset jun 28/Supporting Data - Sheet1.csv"
list = pd.read_csv(path_name)
base_path = "./../../datasets/dataset jun 28/sean & Li 2 exports/"
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

key = 0
for key in result_dict:
    if key==1:
        break
    
    _data= extract_data(result_dict[key])
    # print(_data)
    print("..........")
    # print(_data)
    frames_dir = f'./vids/frames_{key}'
    video_filename=f'./vids/videos/video_{key}.avi'
    create_frames([_data["GP"], _data["BxP"], _data["SP"]],
                   marker_list=['o', 'x', '^'], frame_dir=frames_dir)
    
    key+=1
    create_video_from_frames(frame_dir=frames_dir, fps=30, video_filename=video_filename)