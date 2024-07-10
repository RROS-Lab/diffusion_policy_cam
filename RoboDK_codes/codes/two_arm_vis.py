import numpy as np
# You can also use the new version of the API:
from robodk import robolink    # RoboDK API
from robodk import robomath    # Robot toolbox
RL = robolink.Robolink()
from robodk import *      # RoboDK API
from robolink import *    # Robot toolbox
import re
import pandas as pd
# from ccma import CCMA

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

def extract_data(result_dict, sample_size=8):


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
        # Sampler commented out
        a = data.filter(regex=pattern1).values.astype('float64').tolist()
        # a = sampler(a, sample_size)
        b = data.filter(regex=pattern2).values.astype('float64').tolist()
        # b = sampler(b, sample_size)
        c = data.filter(regex=pattern3).values.astype('float64').tolist()
        # c = sampler(c, sample_size)
        d = data.filter(regex=pattern4).values.astype('float64').tolist()
        # d = sampler(d, sample_size)
        e = data.filter(regex=pattern5).values.astype('float64').tolist()
        # e = sampler(e, sample_size)
        f = data.filter(regex=pattern6).values.astype('float64').tolist()
        # f = sampler(f, sample_size)
        g = data.filter(regex=pattern7).values.astype('float64').tolist()
        # g = sampler(g, sample_size)
        h = data.filter(regex=pattern8).values.astype('float64').tolist()
        # h = sampler(h, sample_size)

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
    # return GXYZ, Gwxyz, SXYZ, Swxyz, BXYZ, Bwxyz, DXYZ, Dwxyz, indexes, GA.tolist()
    return GXYZ, Gwxyz, SXYZ, Swxyz


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


def MuJoCo_2_RoboDK(XYZwxyz):
    '''
    MuJoCo(XYZwxyz) --> ROBODK(XYZwxyz)
    this only takes XYZ and Quaternion and converts from MuJoCo(XYZwxyz) to ROBODK(XYZwxyz)
    -----
    converts meter -> millimeter
    '''
    for i in range(3):
        # print("i -",i)
        # print(XYZwxyz[i]) 
        # XYZwxyz[1]= XYZwxyz[1] + 0.07
        XYZwxyz[i]*=1000
        # ccma = CCMA(w_ma=5, w_cc=3)
        # XYZwxyz[i] = ccma.filter(XYZwxyz[i])
    #     print(XYZwxyz[i]) 
    #     print("************")
    # print(XYZwxyz)
    return XYZwxyz


world_frame = RL.Item("dual_arm_station")
robot0 = RL.Item("robot0")
WORLD_T_ROBOT0 = XYZwxyz_2_Pose(MuJoCo_2_RoboDK([0, 0.880, 0, 0, 0, 0, 0]))
# print(f"WORLD_T_ROBOT0:\n\n{WORLD_T_ROBOT0}\n\********\n\n")

robot1 = RL.Item("robot1")
WORLD_T_ROBOT1 = XYZwxyz_2_Pose(MuJoCo_2_RoboDK([0, -0.470, 0.0, 0, 0, 0, 0]))
# print(f"WORLD_T_ROBOT1:\n\n{WORLD_T_ROBOT1}\n\********\n\n")


_TARGET0_parent = RL.Item("targets_robot0"); RL.Delete(_TARGET0_parent) #Clear Worksapce
_robot0_base = RL.Item("robot0_base")
_TARGET0_parent = RL.AddFrame("targets_robot0", itemparent=_robot0_base); _TARGET0_parent.setVisible(False)
_TARGET0_parent.setPose(eye())

_TARGET1_parent = RL.Item("targets_robot1"); RL.Delete(_TARGET1_parent) #Clear Worksapce
_robot1_base = RL.Item("robot1_base")
_TARGET1_parent = RL.AddFrame("targets_robot1", itemparent=_robot1_base); _TARGET1_parent.setVisible(False)
_TARGET1_parent.setPose(eye())


# path_name = "/home/cam/Documents/test_mujoco_2/Archive/raj_exports/object_bin_take_002.csv"
# path_name = "/home/cam/Documents/test_mujoco_2/Archive/raj_exports/obj_take_004.csv"
# path_name = "/home/cam/Documents/test_mujoco_2/Archive/raj_exports/New_test_002.csv"
# path_name = "/home/cam/Documents/test_mujoco_2/Archive/sean & Li 2 exports/take_215.csv"
path_name = "/home/cam/Documents/test_mujoco_2/Archive/Supporting Data - Sheet1.csv"

list = pd.read_csv(path_name)
# Base path
base_path = "/home/cam/Documents/test_mujoco_2/Archive/sean & Li 2 exports/"
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


joint_robot0 = []
joint_robot1 = []

break_point=0
for key in result_dict:
# for key in reversed(result_dict):
    print(key)
    # if break_point == 20:
    #     break
    key = 131
    GXYZ, Gwxyz, SXYZ, Swxyz = extract_data(result_dict[key])
    count_g = 0
    count_s = 0
    # break_point += 1
    print("Break_point -",break_point)
    cartesian_pose0= []
    cartesian_pose1= []
    if break_point == 1:
        break
    for index, _ in enumerate(zip(GXYZ, Gwxyz, SXYZ, Swxyz)):
        # print(index)
        # print(GXYZ[index])
        # if index<=2840 & index>=2000:
        # if index<=2040 & index>=700:
        # if index%100 == 0:
        current_s = SXYZ[index]

        if index == len(SXYZ) -1:
            next_s = current_s
        else:
            next_s = SXYZ[index+1]

        diff_s = distance(current_s, next_s)
        if diff_s >= 0.003:
            # print("SCOOPER-",[*SXYZ[index] , *Swxyz[index]])
            _TARGET0_index_XYZwxyz = MuJoCo_2_RoboDK([*SXYZ[index] , *Swxyz[index]])
            _TARGET0 = RL.AddTarget(str(f'test-{index}'), itemparent=_TARGET0_parent, itemrobot=robot0)
            # _TARGET0.setPoseAbs(XYZwxyz_2_Pose(_TARGET0_index_XYZwxyz))
            _TARGET0.setPoseAbs(xyzrpw_2_pose(_TARGET0_index_XYZwxyz))
            count_s += 1
            
            # _TARGET1.setPoseAbs(XYZwxyz_2_Pose(_TARGET1_index_XYZwxyz).__mul__(rotx(-1*pi/2)))
            # _TARGET1.setPoseAbs(XYZwxyz_2_Pose(_TARGET1_index_XYZwxyz).__mul__(rotz(-1*pi/2)))
            # _TARGET0.setPoseAbs(XYZwxyz_2_Pose(_TARGET0_index_XYZwxyz).__mul__(roty(-1*pi/2)))

            # WORLD_T_EE1 = XYZwxyz_2_Pose(_TARGET0_index_XYZwxyz).__mul__(roty(1*pi))
            # WORLD_T_EE0 = XYZwxyz_2_Pose(_TARGET0_index_XYZwxyz)
            # ROBOT0_T_EE0 = invH(WORLD_T_ROBOT0).__mul__(WORLD_T_EE0)
            # _TARGET0.setPoseAbs(ROBOT0_T_EE0)
            # cartesian_pose0.append(Pose_2_XYZwxyz(ROBOT0_T_EE0))
            # cartesian_scooper = pd.DataFrame(cartesian_pose0, columns=['x', 'y', 'z', 'w', 'qx', 'qy', 'qz'])
            # desired_order = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'w']
            # cartesian_scooper = cartesian_scooper[desired_order]
            # name_s = result_dict[key]['Path'].split('/')[-1]

        current_g = GXYZ[index]

        if index == len(GXYZ) -1:
            next_g = current_g
        else:
            next_g = GXYZ[index+1]

        diff_g = distance(current_g, next_g)
        if diff_g >= 0.003:
            # print("GRIPPER -",[*GXYZ[index] , *Gwxyz[index]])
            _TARGET1_index_XYZwxyz = MuJoCo_2_RoboDK([*GXYZ[index] , *Gwxyz[index]])
            _TARGET1 = RL.AddTarget(str(f'test-{index}'), itemparent=_TARGET1_parent, itemrobot=robot1)
            # _TARGET1.setPoseAbs(XYZwxyz_2_Pose(_TARGET1_index_XYZwxyz))
            _TARGET1.setPoseAbs(xyzrpw_2_pose(_TARGET1_index_XYZwxyz))
            count_g += 1
            # _TARGET0.setPoseAbs(XYZwxyz_2_Pose(_TARGET0_index_XYZwxyz).__mul__(rotx(-1*pi/2)))
            # _TARGET1.setPoseAbs(XYZwxyz_2_Pose(_TARGET1_index_XYZwxyz).__mul__(rotz(1*pi)))
            # _TARGET0.setPoseAbs(XYZwxyz_2_Pose(_TARGET0_index_XYZwxyz).__mul__(roty(-1*pi/2)))
            # _TARGET0.setPoseAbs(XYZwxyz_2_Pose(_TARGET0_index_XYZwxyz).__mul__(rotx(-1*pi/2)))        
            # WORLD_T_EE1 = XYZwxyz_2_Pose(_TARGET1_index_XYZwxyz).__mul__(roty(1*pi))
            # WORLD_T_EE1 = XYZwxyz_2_Pose(_TARGET1_index_XYZwxyz)
            # ROBOT1_T_EE1 = invH(WORLD_T_ROBOT1).__mul__(WORLD_T_EE1)
            # _TARGET1.setPoseAbs(ROBOT1_T_EE1)
            # cartesian_pose1.append(Pose_2_XYZwxyz(ROBOT1_T_EE1))
            # cartesian_gripper = pd.DataFrame(cartesian_pose1, columns=['x', 'y', 'z', 'w', 'qx', 'qy', 'qz'])
            # desired_order = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'w']
            # cartesian_gripper = cartesian_gripper[desired_order]
            # name_g = result_dict[key]['Path'].split('/')[-1]



        # print(name)

        
    print(f"Total Gripper Targets: {count_g}")
    print(f"Total Scooper Targets: {count_s}")
    # # print("New Trajectory...", cartesian_gripper)
    # cartesian_gripper.to_csv(f'/home/cam/Documents/test_mujoco_2/Archive/Gripper/gripper_pose_{name_g}', index=False)
    # cartesian_scooper.to_csv(f'/home/cam/Documents/test_mujoco_2/Archive/Scooper/scopper_pose_{name_s}', index=False)
    # # input("Press Enter to continue reset Targets...")
    # delete_child_items(RL.Item('targets_robot0'))
    # delete_child_items(RL.Item('targets_robot1'))
    # input("Press Enter to view new trajectory...")

# print(position_1)

if __name__ == "__main__":
    RL.setSimulationSpeed(15)

    for k in cartesian_pose1:
        # print(i)
        # robot1.MoveJ(k)
        pass
    
        # robot1.MoveJ(l)