# import submodules.robomath as rm
import robomath as rm
import numpy as np
from typing import Union

# def quaternion_2_euler(qin : np.ndarray | list) -> np.ndarray:
#     # output is in radians
#     _pose = rm.quaternion_2_pose(qin)
#     _eul = rm.pose_2_xyzrpw(_pose)
#     _eul = np.array(_eul[3:])
#     _eul = _eul*np.pi/180 #convert to radians
    
#     # _eul = convert_theta_to_pi_range(_eul) #normalize the angles from -pi to pi # TODO: comment out
#     return _eul

def TxyzQwxyz_2_Pose(XYZwxyz: Union[list, np.ndarray]) -> rm.Mat:
    '''
    XYZwxyz --> Pose (Homegeneous 4X4 Matrix)
    this converts XYZ Quat to corresponding Homogeneous Matrix
    '''
    _pose_Mat = rm.quaternion_2_pose(XYZwxyz[3:]) #Add Quaternions
    for i in range(3):
        _pose_Mat[i,3] = XYZwxyz[i] #Add XYZ to the last column of the matrix
    return _pose_Mat


def Pose_2_TxyzQwxyz(pose: rm.Mat) -> np.ndarray:
    wxyz=rm.pose_2_quaternion(pose)
    XYZ = [pose[i,3] for i in range(3)]
    return np.array([*XYZ, *wxyz])


def TxyzRxyz_2_TxyzQwxyz(XYZrpy: Union[list, np.ndarray]) -> np.ndarray:
    '''**Rxyz in radians**
    param xyzrpw: [x,y,z,rx,ry,rz] in m/mm and radians
    return: [x,y,z,w,x,y,z] i.e. position and quaternion
    '''
    return Pose_2_TxyzQwxyz(rm.TxyzRxyz_2_Pose(XYZrpy))


def TxyzQwxyz_2_TxyzRxyz(XYZwxyz: Union[list, np.ndarray]) -> np.ndarray:
    '''
    **Rxyz in radians**
    param pose: [x,y,z,w,x,y,z] i.e. position and quaternion
    return: [x,y,z,rx,ry,rz] in m/mm and radians
    '''
    return rm.Pose_2_TxyzRxyz(TxyzQwxyz_2_Pose(XYZwxyz))


def normalize_theta(theta: float) -> float:
    """
    Normalize theta (in radian) to the range [-pi, pi).
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi


def normalize_eulers_method2(angles: np.ndarray) -> np.ndarray:
    """
    Convert a sequence of euler angles (in range -pi to pi) continuous and differentiable to ensure smooth transitions.
    Note: The angles can exceed the range [-pi, pi) after smoothing.
    """
    smooth_angles = angles.copy()
    for i in range(1, len(angles)):
        delta = smooth_angles[i] - smooth_angles[i - 1]
        if np.abs(delta) > np.pi:
            if smooth_angles[i] > smooth_angles[i-1]:
                smooth_angles[i] -= 2 * np.pi
            else:
                smooth_angles[i] += 2 * np.pi
    return smooth_angles


def normalize_eulers(angles: np.ndarray) -> np.ndarray:
    """
    Convert a sequence of euler angles (in range -pi to pi) continuous and differentiable to ensure smooth transitions.
    Note: The angles can exceed the range [-pi, pi) after smoothing.
    """
    rows, cols = angles.shape

    if cols != 3:
        raise ValueError("The input array should have 3 columns for euler angles.")
    for col in range(cols):
        angles[:, col] = np.unwrap(angles[:, col])
    
    return angles
    

''''can stop using these now   
    added new function which is more dynamic then these on in data_filter.py 
'''

def motive_2_robodk_rigidbody(XYZwxyz : Union[list, np.ndarray]) -> np.ndarray:
    """
    Convert motive frame to RoboDK frame.
    """
    X = XYZwxyz[0]; Y = XYZwxyz[1]; Z = XYZwxyz[2]; w = XYZwxyz[3]; x = XYZwxyz[4]; y = XYZwxyz[5]; z = XYZwxyz[6]
    return [Z, X, Y, w, z, x, y]

def motive_2_robodk_marker(XYZ : Union[list, np.ndarray]) -> np.ndarray:
    """
    Convert motive frame to RoboDK frame.
    """
    X = XYZ[0]; Y = XYZ[1]; Z = XYZ[2]
    return [Z, X, Y]

def robodk_2_motive(XYZwxyz : Union[list, np.ndarray]) -> np.ndarray:
    """
    Convert RoboDK quaternions to motive quaternions.
    """
    X = XYZwxyz[0]; Y = XYZwxyz[1]; Z = XYZwxyz[2]; w = XYZwxyz[3]; x = XYZwxyz[4]; y = XYZwxyz[5]; z = XYZwxyz[6]
    # return [Y, Z, X, w, x, y, z]

    return [Z, X, Y, w, z, x, y]



