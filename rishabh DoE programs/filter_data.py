import math
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
# from scipy.spatial.transform import Rotation
from matplotlib.backends.backend_pdf import PdfPages
from robomath import *


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


def normalize_angle(angle):
    """
    Normalize an angle to the range [-pi, pi).
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def smooth_angles_rishabh(angles):
    smooth_angles = angles.copy()
    for i in range(1, len(angles)):
        delta = smooth_angles[i] - smooth_angles[i - 1]
        if np.abs(delta) > pi:
            if smooth_angles[i] > smooth_angles[i-1]:
                smooth_angles[i] -= 2 * pi
            else:
                smooth_angles[i] += 2 * pi
    return smooth_angles

def smooth_angles_chatgpt(angles):
    angles = np.unwrap(angles)
    return angles

def euler_change_rishabh(Gwxyz) : #receives a list of quaternions
    quaternions = np.array(Gwxyz) #reshape the quaternions to a 2D array
    eulers = []

    for _row in range(quaternions.shape[0]):
        # Create a Rotation object from the quaternion
        _quat = quaternions[_row]
        _eul = quaternion_2_euler(_quat)
        eulers.append(_eul)
    
    eulers = np.array(eulers)
    
    rolls = eulers[:,0]
    pitches = eulers[:,1]
    yaws = eulers[:,2]

    smooth_rolls = smooth_angles_rishabh(rolls)
    smooth_pitches = smooth_angles_rishabh(pitches)
    smooth_yaws = smooth_angles_rishabh(yaws)

    smooth_eulers = np.array([smooth_rolls, smooth_pitches, smooth_yaws]).T

    return smooth_eulers


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
    return smooth_eulers



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

def delete_child_items(item):
    # Retrieve all child items inside the parent item
    child_items = item.Childs()

    # Iterate through each child item and delete it
    for child in child_items:
        child.Delete()

    print(f"All child items inside '{item.Name()}' have been deleted.")


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


def create_subplots(*datasets, **kwargs):
    subplot_count = len(datasets)
    plt.figure(figsize=(20, 8))
    layouts = kwargs.get('layouts')
    titles = kwargs.get('title')
    sup_title = kwargs.get('sup_title')

    if not titles:
        titles = [f'{i+1}' for i in range(subplot_count)]

    if not layouts:
        layouts = [(1, subplot_count, i+1) for i in range(subplot_count)]

    if not sup_title:
        sup_title = ''

    y_lims = ()
    # Determine the common y-axis limits
    # min_y = np.array(datasets).min() #commented out
    # max_y = np.array(datasets).max() #commented out)
    # y_lims = (min_y, max_y) #commented out

    for i, data in enumerate(datasets):
        plt.suptitle(sup_title)
        plt.subplot(*layouts[i])
        # data = [[_i[0],_i[1]] for _i in data] #commented out
        plt.plot(data)
        plt.legend(['X', 'Y', 'Z']) #commented out

        plt.title(titles[i])
        if y_lims:
            plt.ylim(*y_lims)
        plt.tight_layout()


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


with PdfPages('experiment_results.pdf') as pdf:
    for key in result_dict:
        _data= extract_data(result_dict[key])
        # euler_Gwxyz = euler_change_old(Gwxyz)  # Convert quaternions to Euler angles
        # euler_Gwxyz = Gwxyz
        # {'GR', 'GP', 'SR', 'SP', 'BxR', 'BxP', 'BuR', 'BuP'}
        create_subplots(_data['GP'],
                        _data['GR'],
                        _data['SP'],
                        _data['SR'],
                        title=['Gripper Pos', 'Gripper Rot', 'Scooper Pos','Scooper Rot'],
                        layouts = [(2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4)],
                        sup_title = f"take_{key}")

        print(f"creating subplots for {key}")  
        pdf.savefig()
        plt.close()



# for key in result_dict:
#     _data= extract_data(result_dict[key])
#     print(_data)
#     # create_xy_video(_data['GP'], _data['BxP'], _data['SP'], fps=30, video_filename=f'videos/output_{key}.avi', marker_list=['o', 'x', '*'])
