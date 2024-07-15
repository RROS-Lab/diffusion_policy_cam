import h5py
import numpy as np
# You can also use the new version of the API:
from robodk import robolink    # RoboDK API
from robodk import robomath    # Robot toolbox
RL = robolink.Robolink()
from robodk import *      # RoboDK API
from robolink import *    # Robot toolbox



def XYZwxyz_2_Pose(XYZwxyz):
    '''
    XYZwxyz --> Pose (Homegeneous 4X4 Matrix)
    this converts XYZ Quat to corresponding Homogeneous Matrix
    '''
    _pose_matrix_without_XYZ = quaternion_2_pose(XYZwxyz[3:])
    for i in range(3):
        _pose_matrix_without_XYZ[i,3] = XYZwxyz[i]
    return _pose_matrix_without_XYZ

def MuJoCo_2_RoboDK(XYZwxyz):
    '''
    MuJoCo(XYZwxyz) --> ROBODK(XYZwxyz)
    this only takes XYZ and Quaternion and converts from MuJoCo(XYZwxyz) to ROBODK(XYZwxyz)
    -----
    converts meter -> millimeter
    '''
    for i in range(3): XYZwxyz[i]*=1000

    return XYZwxyz



robot0 = RL.Item("robot0")
WORLD_T_ROBOT0 = XYZwxyz_2_Pose(MuJoCo_2_RoboDK([0, -0.810, 0.8, 0.707107, 0, 0, 0.707107]))
print(f"WORLD_T_ROBOT0:\n\n{WORLD_T_ROBOT0}\n\********\n\n")

robot1 = RL.Item("robot1")
WORLD_T_ROBOT1 = XYZwxyz_2_Pose(MuJoCo_2_RoboDK([0, 0.810, 0.8, 0.707107, 0, 0, -0.707107]))



# # Notify user:
# print('To edit this program:\nright click on the Python program, then, select "Edit Python script"')

# # Program example:

dataset_path = "/home/rishabh/Desktop/dual kuka stations/datasets/low_dim.hdf5"
with h5py.File(dataset_path) as f:
    data_group = f['/data']
    
    for t in range(1):
        # Extract model XML from the first demonstration
        first_demo_key = list(data_group.keys())[t]
        first_demo_grp = data_group[first_demo_key]

        # Extract observations, actions, rewards, dones for the first demonstration
        actions = first_demo_grp['actions'][:]
        # print("actions =",actions[0].shape)
        dones = first_demo_grp['dones']
        next_obs = first_demo_grp['next_obs']
        obs = first_demo_grp['obs']
        print("obs =",obs.keys())
        rewards = first_demo_grp['rewards']
        states = first_demo_grp['states']
        # print("states =",first_demo_grp['states'])

        object_obs = obs['object'][:]
        robot0_eef_quat = obs['robot0_eef_quat'][:]
        robot0_eef_pos = obs['robot0_eef_pos'][:]
        robot0_gripper_qpos = obs['robot0_gripper_qpos'][:]

        robot1_eef_quat = obs['robot1_eef_quat'][:]
        robot1_eef_pos = obs['robot1_eef_pos'][:]
        robot1_gripper_qpos = obs['robot1_gripper_qpos'][:]

        # print("Object observations shape:", object_obs[0].shape)
        # print("Robot end-effector quaternion shape:", robot0_eef_quat[0].shape)
        # print("Robot end-effector position shape:", robot0_eef_pos[0].shape)
        # print("Robot gripper qpos shape:", robot0_gripper_qpos[0].shape)

        # print("Robot end-effector quaternion shape:", robot1_eef_quat[0].shape)
        # print("Robot end-effector position shape:", robot1_eef_pos[0].shape)
        # print("Robot gripper qpos shape:", robot1_gripper_qpos[0].shape)

        
        joint_robot0 = []
        cartesian_pose0= []

        break_point=0
        for index, pos, quat in zip(range(len(robot0_eef_pos)), robot0_eef_pos, robot0_eef_quat):            
            # if index%5 !=0:continue
            # if break_point==1:
            #     break
            # break_point+=1

            # WORLD_T_EE0 = XYZwxyz_2_Pose(MuJoCo_2_RoboDK(pos + quat))

            WORLD_T_EE0 = XYZwxyz_2_Pose(MuJoCo_2_RoboDK([*pos , *quat])).__mul__(rotx(1*pi/2))

            ROBOT0_T_EE0 = invH(WORLD_T_ROBOT0).__mul__(WORLD_T_EE0)
            
            # import pdb; pdb.set_trace()

            # print(f"WORLD_T_EE0:\n\n{WORLD_T_EE0}\n\********\n\n")
            # print(f"ROBOT0_T_EE0:\n\n{ROBOT0_T_EE0}\n\********\n\n")

            # JOINTS_ROBOT0 = robot0.SolveIK(ROBOT0_T_EE0)
            
            # print(JOINTS_ROBOT0.__format__)
            # Check if all elements of JOINTS_ROBOT0 are zeros
            cartesian_pose0.append(ROBOT0_T_EE0)

            '''
            check = JOINTS_ROBOT0.toNumpy()
            if not np.all(check == 0):
            #     # Append JOINTS_ROBOT0 to joint_robot0
                joint_robot0.append(JOINTS_ROBOT0)
                cartesian_pose0.append(ROBOT0_T_EE0)
            '''

        # print(joint_robot0[0])

        joint_robot1 = []
        cartesian_pose1= []

        break_point=0
        for index, pos, quat in zip(range(len(robot1_eef_pos)), robot1_eef_pos, robot1_eef_quat):            
            # if index%5 !=0:continue
            # if break_point==1:
            #     break
            # break_point+=1

            # WORLD_T_EE0 = XYZwxyz_2_Pose(MuJoCo_2_RoboDK(pos + quat))

            WORLD_T_EE1 = XYZwxyz_2_Pose(MuJoCo_2_RoboDK([*pos , *quat])).__mul__(rotx(1*pi/2))

            ROBOT1_T_EE1 = invH(WORLD_T_ROBOT1).__mul__(WORLD_T_EE1)

            cartesian_pose1.append(ROBOT1_T_EE1)




# print(position_1)

if __name__ == "__main__":
    RL.setSimulationSpeed(15)
    # print(WORLD_T_ROBOT1)

    # target1 = RL.Item("Target 1")
    # robot1.MoveJ(target1.Pose())

    # print(target1.Pose())

    
    # for i, j in zip(joint_robot0, joint_robot1):
    #     # print(i)
    #     robot0.MoveJ(i)
    #     robot1.MoveJ(j)

    for k, l in zip(cartesian_pose0, cartesian_pose1):
        # print(i)
        robot0.MoveJ(k)
    
        # robot1.MoveJ(l)
