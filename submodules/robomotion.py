from robodk import robolink    # RoboDK API
from robodk import robomath as rm   # Robot toolbox
RDK = robolink.Robolink()
import csv
# Forward and backwards compatible use of the RoboDK API:
# Remove these 2 lines to follow python programming guidelines
from robodk import *      # RoboDK API
from robolink import *    # Robot toolbox
import pandas as pd
import time
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import pandas as pd
from scipy.interpolate import CubicSpline

def lerp(start, end, t):
    return (1 - t) * start + t * end

def slerp(start, end, t):
    """
    Spherical Linear Interpolation (SLERP) between two quaternions.
    """
    dot = np.dot(start, end)

    # If the dot product is negative, slerp won't take the shorter path.
    # Note that v1 and -v1 are equivalent when the negations can be absorbed by the quaternion multiplication.
    if dot < 0.0:
        end = -end
        dot = -dot

    # Compute the angle between the start and end quaternions
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    # If the quaternions are very close, then just perform linear interpolation
    if sin_theta_0 < 1e-6:
        return (1.0 - t) * start + t * end

    # Compute the angle at t
    theta = theta_0 * t
    sin_theta = np.sin(theta)

    # Compute s0 and s1
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return s0 * start + s1 * end

def interpolate_pose(pose1, pose2, t):
    # Extract the rotation matrices and translation vectors
    rot1, trans1 = pose1[:3, :3], pose1[:3, 3]
    rot2, trans2 = pose2[:3, :3], pose2[:3, 3]

    # Convert rotation matrices to scipy Rotation objects
    r1, r2 = R.from_matrix([rot1]), R.from_matrix([rot2])

    # Perform SLERP on the rotation matrices
    slerped_rot = R.slerp(0, 1, [r1, r2])(t)

    # Linearly interpolate the translation vectors
    lerp_trans = (1 - t) * trans1 + t * trans2

    # Construct the interpolated pose matrix
    interpolated_pose = np.eye(4)
    interpolated_pose[:3, :3] = slerped_rot.as_matrix()[0]
    interpolated_pose[:3, 3] = lerp_trans

    return interpolated_pose

# Cubic spline interpolation for position
def cubic_spline_interpolation(waypoints, steps=100):
    x_vals = np.linspace(0, 1, len(waypoints))
    cs_x = CubicSpline(x_vals, [wp[0] for wp in waypoints])
    cs_y = CubicSpline(x_vals, [wp[1] for wp in waypoints])
    cs_z = CubicSpline(x_vals, [wp[2] for wp in waypoints])
    t_vals = np.linspace(0, 1, steps)
    return np.vstack((cs_x(t_vals), cs_y(t_vals), cs_z(t_vals))).T

def point_2_point(start, stop, steps=100):
    positions = []
    quaternions = []
    for t in np.linspace(0, 1, steps):
        alpha = t
        pos = lerp(start[:3], stop[:3], alpha)
        quat = slerp(start[3:], stop[3:], alpha)
        positions.append(pos)
        quaternions.append(quat)
    return np.concatenate([np.array(positions), np.array(quaternions)], axis=1)

def smooth_trajectory_straight(waypoint1, waypoint2, waypoint3, steps=100):
    positions = []
    quaternions = []
    for t in np.linspace(0, 1, steps):
        if t < 0.5:
            alpha = 2 * t
            pos = lerp(waypoint1[:3], waypoint2[:3], alpha)
            quat = slerp(waypoint1[3:], waypoint2[3:], alpha)
        else:
            alpha = 2 * (t - 0.5)
            pos = lerp(waypoint2[:3], waypoint3[:3], alpha)
            quat = slerp(waypoint2[3:], waypoint3[3:], alpha)
        positions.append(pos)
        quaternions.append(quat)
    return np.concatenate([np.array(positions), np.array(quaternions)], axis=1)
# Generate smooth trajectory

def smooth_trajectory_cubic(waypoints, steps=100):
    positions = cubic_spline_interpolation([wp[:3] for wp in waypoints], steps)
    quaternions = []

    for t in np.linspace(0, 1, steps):
        if t < 0.5:
            alpha = 2 * t
            quat = slerp(waypoints[0][3:], waypoints[1][3:], alpha)
        else:
            alpha = 2 * (t - 0.5)
            quat = slerp(waypoints[1][3:], waypoints[2][3:], alpha)
        quaternions.append(quat)

    return np.concatenate([np.array(positions), np.array(quaternions)], axis=1)


def generate_csv_trajectory(position_trajectory, orientation_trajectory, csv_file_path):
    combined_trajectory = np.hstack((position_trajectory, orientation_trajectory))
    df = pd.DataFrame(combined_trajectory, columns=['X', 'Y', 'Z', 'x', 'y', 'z', 'w'])
    df.to_csv(csv_file_path, index=False)
    return csv_file_path



def get_equally_spaced_items(lst, n):
    if n > len(lst) or n < 2: return "Error: n must be between 2 and the length of the list"
    step = (len(lst) - 1) / (n - 1)
    result = [lst[int(round(step * i))] for i in range(n)]
    return result


def generate_waypoints_from_targets(targets):
    target_xyz = lambda target: np.array(Pose_2_KUKA(target.Pose())[0:3])/1000
    target_quat = lambda target: np.array(pose_2_quaternion(target.Pose()))
    waypoints = [np.concatenate([target_xyz(target),target_quat(target)]) for target in targets]
    return np.vstack(waypoints)

def generate_geometric_path_from_waypoints(waypoints, steps = 100, kind="cubic"):
    if len(waypoints) > 2:
        return smooth_trajectory_cubic(waypoints, steps=steps)
    if len(waypoints) == 2:
        return point_2_point(waypoints[0], waypoints[1])
    

def plot_path(ax, path):
    ax.plot(path[:, 0], path[:, 1], path[:, 2])

def plot_nodes(ax, path):
    ax.scatter(path[:, 0], path[:, 1], path[:, 2], c='black', marker='o')

def plot_coordinate_frame(ax, frame, size=0.1, linewidth = 1):
    position = frame[0:3]
    quaternion = frame[3:]
    rotation_matrix = quaternion_2_pose(quaternion)
    x_axis = rotation_matrix[:, 0] * size
    y_axis = rotation_matrix[:, 1] * size
    z_axis = rotation_matrix[:, 2] * size
    ax.quiver(position[0], position[1], position[2], x_axis[0], x_axis[1], x_axis[2], color='r', linewidth=linewidth)
    ax.quiver(position[0], position[1], position[2], y_axis[0], y_axis[1], y_axis[2], color='g', linewidth=linewidth)
    ax.quiver(position[0], position[1], position[2], z_axis[0], z_axis[1], z_axis[2], color='b', linewidth=linewidth)

def plot_geometric_path(ax, geometric_path, size = 0.1, linewidth = 1):
    for frame in geometric_path:
        plot_coordinate_frame(ax, frame, size=size, linewidth=linewidth)

def plot_motion_xyz(ax, geometric_path):
    ax.grid(True)
    ax.set_facecolor('lightgray')
    ax.set_title('Path')
    ax.plot(geometric_path[:, 0], color='blue')
    ax.plot(geometric_path[:, 1], color='red')
    ax.plot(geometric_path[:, 2], color='green')
    ax.legend(['X', 'Y', 'Z'])

def plot_motion_rpy(ax, geometric_path):
    ax.grid(True)
    ax.set_facecolor('lightgray')
    euler_path = []
    for frame in geometric_path:
        xyz_rpy = Pose_2_KUKA(quaternion_2_pose(frame[3:]))
        euler_path.append(xyz_rpy)
    euler_path = np.vstack(euler_path)
    ax.set_title('YPR')
    ax.plot(euler_path[:, 3], color='orange')
    ax.plot(euler_path[:, 4], color='magenta')
    ax.plot(euler_path[:, 5], color='brown')
    ax.legend(['Y', 'P', 'R'])

def plot_motion_quat(ax, geometric_path):
    ax.grid(True)
    ax.set_facecolor('lightgray')
    ax.set_title('Quaternion')
    ax.plot(geometric_path[:, 3], color='lightgreen')
    ax.plot(geometric_path[:, 4], color='orange')
    ax.plot(geometric_path[:, 5], color='blue')
    ax.plot(geometric_path[:, 6], color='black')
    ax.legend(['w', 'x', 'y', 'z'])


def generate_sunrise_commands_from_geometric_path(geometric_path, kind="spline", segmentID="", pause_in_seconds=0):
    frame_definition_block = "\n"
    if segmentID != "":
        segmentID = f"_ID{segmentID}"
    for index, frame in enumerate(geometric_path):
        position = frame[0:3]
        quaternion = frame[3:]
        position_in_mm = np.round(position*1000,3)
        euler_deg = Pose_2_KUKA(quaternion_2_pose(quaternion))[3:]
        # euler_ypr = euler_deg.reverse()
        # =============================================
        # =        Euler rpy has to be converted to YPR comment block=
        # =============================================
        euler_rad = [round(j* pi/180,3) for j in euler_deg]
        frame_definition_block+=f"Frame FrameLIN{segmentID}_{index+1} = new Frame{*position_in_mm, *euler_rad};\n"

    move_definition_block = "\n"
    modifier={"spline": "spl", "linear":"lin"}
    delimiter = ","

    for index, frame in enumerate(geometric_path):
        if index == len(geometric_path) - 1: delimiter = ""
        move_definition_block+=f"\t{modifier[kind]}(FrameLIN{segmentID}_{index+1}).setCartVelocity(50).setCartAcceleration(50){delimiter}\n"

    output_block = f'''
                    {frame_definition_block}
move_curve = new Spline({move_definition_block});
TOOL.move(move_curve);
                    '''
    # print(output_block)

    if pause_in_seconds !=0:
        pause_in_ms = pause_in_seconds*1000
        output_block += '''
try {{
    Thread.sleep({0}); // Pause for {0} milliseconds
}} catch (InterruptedException e) {{
    getLogger().info("Pause Skipped");
}}
'''.format(pause_in_ms)

    print(output_block)

def run_on_KUKA(geometric_path, speed = 10):
    KUKA = RDK.Item('KUKA LBR iiwa 7 R800')
    row = {}

    for frame in geometric_path:
        X = frame[0]; Y = frame[1]; Z = frame[2]
        w = frame[3]; x = frame[4]; y = frame[5]; z = frame[6]
        pose = rm.transl(X*1000, Y*1000, Z*1000)* quaternion_2_pose([w, x, y, z])
        # KUKA.setPose(pose)
        KUKA.MoveL(pose)
        joints = np.round(np.array(KUKA.Joints()).flatten(),1)
        # print(joints)
        time.sleep(1/speed)
    return KUKA


def set_3D_plot_axis_and_labels(ax):
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1]) # Set the scale
    ax.set_xlim([0,0.5])
    ax.set_ylim([0,0.5])
    ax.set_zlim([0,0.5])
    # ax.legend()
    ax.set_title('3D Spline Trajectory')

def single_segment_motion_from_targets(ax, targets, kind="spline", density=10, plotting=True, simulate=False, segmentID="", pause_in_seconds=0):
    waypoints = generate_waypoints_from_targets(targets)
    geometric_path_spline = generate_geometric_path_from_waypoints(waypoints)
    n_geometric_path_frames = get_equally_spaced_items(lst=geometric_path_spline, n=density)
    generate_sunrise_commands_from_geometric_path(n_geometric_path_frames, kind=kind, segmentID=segmentID, pause_in_seconds=pause_in_seconds)

    if simulate:
        run_on_KUKA(geometric_path_spline, speed=200)

    if plotting:
        plot_path(ax, geometric_path_spline)
        plot_nodes(ax, waypoints)
        plot_geometric_path(ax, waypoints, size=0.05, linewidth=2)
        plot_geometric_path(ax, n_geometric_path_frames, size=0.04, linewidth=1)
        # plot_motion_xyz_quat([ax[1],ax[2]], geometric_path_spline)

    return geometric_path_spline



def multiple_segments_motion_from_targets(*segments_with_targets_and_types_and_density, plotting=True, simulate=False):
    fig = plt.figure(figsize=(15, 8))
    fig.subplots_adjust(left=0.036, bottom=0.042, right=0.98, top=0.96, wspace=0, hspace=0.233)

    ax1 = fig.add_subplot(121, projection='3d')
    set_3D_plot_axis_and_labels(ax1)
    ax2 = fig.add_subplot(333)
    ax3 = fig.add_subplot(336)
    ax4 = fig.add_subplot(339)

    # ax  = [ax1, ax2, ax3]

    spline_segments_list = []
    for segmentID, (segment_targets, kind, density) in enumerate(segments_with_targets_and_types_and_density):
        spline_segment = single_segment_motion_from_targets(ax1, segment_targets, kind, density, plotting=plotting, simulate=simulate, segmentID=segmentID)
        spline_segments_list.append(spline_segment)

    full_spline = np.vstack(spline_segments_list)
    # plot_motion_xyz_quat([ax[1],ax[2]], full_spline)
    plot_motion_xyz(ax2, full_spline)
    plot_motion_rpy(ax3, full_spline)
    plot_motion_quat(ax4, full_spline)
    return full_spline


if __name__ == "__main__":
    simulate=False; plotting=True
    # plotting=False; simulate=False
    pickup_group = 1
    spline_group = 1
    place_group  = 1 


    RDK.setSimulationSpeed(200)


    base = RDK.Item("base") 
    # bag = RDK.Item("bag1")
    gripper = RDK.Item("gp")
    KUKA = RDK.Item('KUKA LBR iiwa 7 R800')

    # bag.setParent(base)
    # bag_initial_pose = rm.xyzrpw_2_pose([   100.000000,   600.000000,    40.000000,  -180.000000,     0.000000,    60.000000 ])
    # bag.setPose(bag_initial_pose)
    
    

    pickup_targets   = sorted(RDK.Item(f"pickup_{pickup_group}").Childs(), key = lambda i: i.Name())
    spline_targets   = sorted(RDK.Item(f"spline_{spline_group}").Childs(), key = lambda i: i.Name())
    # preplace_targets = sorted(RDK.Item(f"preplace_{motion}").Childs(), key= lambda i: i.Name())
    place_targets    = sorted(RDK.Item(f"place_{place_group}").Childs(), key = lambda i: i.Name())

    prepick_segment_targets = [pickup_targets[1], pickup_targets[0]]
    
    spline_segments_targets = [
                               pickup_targets[1],  #up
                               spline_targets[-1], # adjustable last target in spline groups
                               spline_targets[0] #
                               ]
    
    place_segments_targets  = [spline_targets[0], place_targets[0]]


    # homing = [preplace_targets[0], preplace_targets[-1], pickup_targets[1]]


    fig = plt.figure(figsize=(15, 8))
    fig.subplots_adjust(left=0.036, bottom=0.042, right=0.98, top=0.96, wspace=0, hspace=0.233)

    ax1 = fig.add_subplot(121, projection='3d')
    set_3D_plot_axis_and_labels(ax1)
    ax2 = fig.add_subplot(333)
    ax3 = fig.add_subplot(336)
    ax4 = fig.add_subplot(339)

    # seg1 = single_segment_motion_from_targets(ax1, prepick_segment_targets, "linear", 3, plotting=plotting, simulate=simulate, segmentID=1, pause_in_seconds=0)

    source_down = single_segment_motion_from_targets(ax1, prepick_segment_targets, "linear", 2, plotting=plotting, simulate=simulate, segmentID=1, pause_in_seconds=1)
    source_up = single_segment_motion_from_targets(ax1, reversed(prepick_segment_targets), "linear", 2, plotting=plotting, simulate=simulate, segmentID=2, pause_in_seconds=1)


    # bag_gp_rel_pose = rm.KUKA_2_Pose([ 0,   0,    0,   180.000000,     0.000000,  -180.000000 ]) # bag.setParentStatic(gripper) # bag.setPose(bag_gp_rel_pose)

    
    spline = single_segment_motion_from_targets(ax1, spline_segments_targets, "spline", 4, plotting=plotting, simulate=simulate, segmentID=3, pause_in_seconds=0)
    
    # seg3 = single_segment_motion_from_targets(ax1, place_segments_targets, "linear", 3, plotting=plotting, simulate=simulate, segmentID=3, pause_in_seconds=0)
    
    target_down = single_segment_motion_from_targets(ax1, place_segments_targets, "linear", 3, plotting=plotting, simulate=simulate, segmentID=4, pause_in_seconds=1)
    target_up = single_segment_motion_from_targets(ax1, reversed(place_segments_targets), "linear", 3, plotting=plotting, simulate=simulate, segmentID=5, pause_in_seconds=0)
    
    spline_reverse = single_segment_motion_from_targets(ax1, reversed(spline_segments_targets), "spline", 3, plotting=plotting, simulate=simulate, segmentID=6, pause_in_seconds=0)


    # homing = single_segment_motion_from_targets(ax1, homing, "linear", 3, plotting=plotting, simulate=simulate, segmentID=3, pause_in_seconds=0)
    # bag.setParentStatic(base) # RDK.RunProgram('Prog4') # bag.setPose(bag_initial_pose)

    # seg4 = single_segment_motion_from_targets(ax3, homing, "spline", 10, plotting=False, simulate=True, segmentID=3)
    # KUKA.MoveL(preplace_targets[-1])
    # time.sleep(10)
    # KUKA.MoveL(pickup_targets[1])
    
    

    # full_spline = multiple_segments_motion_from_targets(
    #                                       (prepick_segment_targets, "linear", 3),
    #                                       (spline_segments_targets, "spline", 10),
    #                                       (place_segments_targets, "linear", 3),
    #                                       simulate=True, plotting=False
    #                                       )

    if plotting:
        plt.show()
