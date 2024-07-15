import numpy as np
# You can also use the new version of the API:
from robodk import robolink    # RoboDK API
from robodk import robomath    # Robot toolbox
RL = robolink.Robolink()
from robodk import *      # RoboDK API
from robolink import *    # Robot toolbox
import re
import pandas as pd



def XYZwxyz_2_Pose(XYZwxyz):
    '''
    XYZwxyz --> Pose (Homegeneous 4X4 Matrix)
    this converts XYZ Quat to corresponding Homogeneous Matrix
    '''
    _pose_matrix_without_XYZ = quaternion_2_pose(XYZwxyz[3:])
    for i in range(3):
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
        print("i -",i)
        print(XYZwxyz[i]) 
        XYZwxyz[i]*=1000
    # _new_pose = XYZwxyz_2_Pose(XYZwxyz)*rotx(pi/2)
    # _newXYZwxyz = Pose_2_XYZwxyz(_new_pose)
    return XYZwxyz



robot0 = RL.Item("robot0")
WORLD_T_ROBOT0 = XYZwxyz_2_Pose(MuJoCo_2_RoboDK([0, -0.810, 0.912, 0.707107, 0, 0, 0.707107]))
# print(f"WORLD_T_ROBOT0:\n\n{WORLD_T_ROBOT0}\n\********\n\n")

robot1 = RL.Item("robot1")
WORLD_T_ROBOT1 = XYZwxyz_2_Pose(MuJoCo_2_RoboDK([0, 0.810, 0.912, 0.707107, 0, 0, -0.707107]))
# print(f"WORLD_T_ROBOT1:\n\n{WORLD_T_ROBOT1}\n\********\n\n")


_TARGET0_parent = RL.Item("targets_robot0"); RL.Delete(_TARGET0_parent) #Clear Worksapce
_robot0_base = RL.Item("robot0_base")
_TARGET0_parent = RL.AddFrame("targets_robot0", itemparent=_robot0_base); _TARGET0_parent.setVisible(False)
_TARGET0_parent.setPose(eye())

_TARGET1_parent = RL.Item("targets_robot1"); RL.Delete(_TARGET1_parent) #Clear Worksapce
_robot1_base = RL.Item("robot1_base")
_TARGET1_parent = RL.AddFrame("targets_robot1", itemparent=_robot1_base); _TARGET1_parent.setVisible(False)
_TARGET1_parent.setPose(eye())


path_name = "/home/cam/Documents/test_mujoco_2/Archive/take001_005.csv"

data  = pd.read_csv(path_name)
data = data.reset_index(drop=True)

row1 = data.iloc[1]
row2 = data.iloc[3]
row3 = data.iloc[4]

combined_values = []
for a, b, c in zip(row1, row2, row3):
    combined_values.append(str(a) + '_' + str(b) + '_' + str(c))

data.iloc[6] = combined_values
data = data.drop(range(6))
data = data.drop(data.columns[:2], axis=1)
data = data.reset_index(drop=True)
data = data.dropna()
data = data.reset_index(drop=True)
data.columns = data.iloc[0]
data = data.drop(0)
data = data.reset_index(drop=True)

# Regular expression pattern to match columns starting with 'gripper_1_Rotation'
pattern1 = re.compile(r'gripper_1_Rotation')
pattern2 = re.compile(r'gripper_1_Position')
pattern3 = re.compile(r'scooper_2_Rotation')
pattern4 = re.compile(r'scooper_2_Position')

# Filter columns using regex pattern and extract values into a list
Gwxyz = data.filter(regex=pattern1).values.tolist()
GXYZ = data.filter(regex=pattern2).values.tolist()
Swxyz = data.filter(regex=pattern3).values.tolist()
SXYZ = data.filter(regex=pattern4).values.tolist()
    

joint_robot0 = []
cartesian_pose0= []


break_point=0
        


# _TARGET0_index_XYZwxyz = MuJoCo_2_RoboDK([*SXYZ , *Swxyz])
# _TARGET0 = RL.AddTarget(str(index), itemparent=_TARGET0_parent)
# _TARGET0.setPoseAbs(XYZwxyz_2_Pose(_TARGET0_index_XYZwxyz))
for index, _ in enumerate(zip(GXYZ, Gwxyz)):
        # print(index)
    # print(GXYZ[index])
    _TARGET1_index_XYZwxyz = MuJoCo_2_RoboDK([GXYZ[index] , Gwxyz[index]])
    _TARGET1 = RL.AddTarget(str('test'), itemparent=_TARGET0_parent)
    _TARGET1.setPoseAbs(XYZwxyz_2_Pose(_TARGET1_index_XYZwxyz))

# print(position_1)

if __name__ == "__main__":
    RL.setSimulationSpeed(15)

    for k in cartesian_pose0:
        # print(i)
        # robot0.MoveJ(k)
        pass
    
        # robot1.MoveJ(l)
