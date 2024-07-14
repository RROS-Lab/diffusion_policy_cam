from matplotlib import pyplot as plt
import submodules.robomath as rm
import submodules.robomath_addon as rma
import numpy as np
import pandas as pd

class PlotTraj3D(object):
    def __init__(self):
        print("3D_traj_plot.py is being run directly")
        self.fig = plt.figure(figsize=(15, 8))
        self.fig.subplots_adjust(left=0.036, bottom=0.042, right=0.98, top=0.96, wspace=0, hspace=0.233)

    
    def add_subplot(self, nrows: int, ncols:int, index:int, projection:str='rectilinear') -> plt.Axes:
        _ax = self.fig.add_subplot(nrows, ncols, index, projection=projection)
        return _ax

    @staticmethod
    def plot_path(ax:plt.Axes, path: np.ndarray) -> None:
        ax.plot(path[:, 0], path[:, 1], path[:, 2])

    @staticmethod
    def plot_nodes(ax:plt.Axes, path: np.ndarray) -> None:
        ax.scatter(path[:, 0], path[:, 1], path[:, 2], c='black', marker='o')

    
    @staticmethod
    def plot_coordinate_frame(ax:plt.Axes, XYZwxyz: np.ndarray[7], size:float=0.1, linewidth:float=1) -> None:
        #frame is position + quaternion of size 7
        position = XYZwxyz[0:3]
        quaternion = XYZwxyz[3:]
        rotation_matrix = rm.quaternion_2_pose(quaternion)
        x_axis = rotation_matrix[:, 0] * size
        y_axis = rotation_matrix[:, 1] * size
        z_axis = rotation_matrix[:, 2] * size
        ax.quiver(position[0], position[1], position[2], x_axis[0], x_axis[1], x_axis[2], color='r', linewidth=linewidth)
        ax.quiver(position[0], position[1], position[2], y_axis[0], y_axis[1], y_axis[2], color='g', linewidth=linewidth)
        ax.quiver(position[0], position[1], position[2], z_axis[0], z_axis[1], z_axis[2], color='b', linewidth=linewidth)

    @staticmethod
    def plot_geometric_path(ax:plt.Axes, geometric_path: np.ndarray[np.ndarray[7]], size:float=0.1, linewidth:float=1) -> None:
        #lsit of frames
        for frame in geometric_path:
            PlotTraj3D.plot_coordinate_frame(ax, frame, size, linewidth)

    @staticmethod
    def plot_motion_xyz(ax: plt.Axes, geometric_path: np.ndarray[np.ndarray[7]]) -> None:
        ax.grid(True)
        ax.set_facecolor('lightgray')
        ax.set_title('Path')
        ax.plot(geometric_path[:, 0], color='blue')
        ax.plot(geometric_path[:, 1], color='red')
        ax.plot(geometric_path[:, 2], color='green')
        ax.legend(['X', 'Y', 'Z'])

    @staticmethod
    def plot_motion_rpy(ax: plt.Axes, geometric_path: np.ndarray[np.ndarray[7]]) -> None:
        ax.grid(True)
        ax.set_facecolor('lightgray')
        euler_path = []
        for frame in geometric_path:
            xyz_rpy = rm.Pose_2_KUKA(rm.quaternion_2_pose(frame[3:])) #TODO: check if this is correct conversion pose_2_xyzrpw??
            euler_path.append(xyz_rpy)
        euler_path = np.vstack(euler_path)
        ax.set_title('YPR')
        ax.plot(euler_path[:, 3], color='orange')
        ax.plot(euler_path[:, 4], color='magenta')
        ax.plot(euler_path[:, 5], color='brown')
        ax.legend(['Y', 'P', 'R'])

    @staticmethod
    def plot_motion_quat(ax: plt.Axes, geometric_path: np.ndarray[np.ndarray[7]]) -> None:
        ax.grid(True)
        ax.set_facecolor('lightgray')
        ax.set_title('Quaternion')
        ax.plot(geometric_path[:, 3], color='lightgreen')
        ax.plot(geometric_path[:, 4], color='orange')
        ax.plot(geometric_path[:, 5], color='blue')
        ax.plot(geometric_path[:, 6], color='black')
        ax.legend(['w', 'x', 'y', 'z'])

    @staticmethod
    def set_3D_plot_axis_and_labels(ax:plt.Axes) -> None:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1]) # Set the scale
        ax.set_xlim([0,0.5])
        ax.set_ylim([0,0.5])
        ax.set_zlim([0,0.5])
        # ax.legend()
        ax.set_title('3D Spline Trajectory')


    def main(self, traj: np.ndarray[np.ndarray[7]], density:int=10) -> None:
        path = traj[:, 0:3]
        # nodes and frames are equally spaced points on the path
        nodes = path[::density]
        frames = traj[::density]
        ax1 = self.add_subplot(1, 2, 1, projection='3d')
        self.set_3D_plot_axis_and_labels(ax1)
        self.plot_path(ax1, path)
        self.plot_nodes(ax1, nodes)
        self.plot_geometric_path(ax1, frames, size=0.04, linewidth=1)

        ax2 = self.add_subplot(3, 3, 3)
        ax3 = self.add_subplot(3, 3, 6)
        ax4 = self.add_subplot(3, 3, 9)

        self.plot_motion_xyz(ax2, traj)
        self.plot_motion_rpy(ax3, traj)
        self.plot_motion_quat(ax4, traj)

        plt.show()
    
