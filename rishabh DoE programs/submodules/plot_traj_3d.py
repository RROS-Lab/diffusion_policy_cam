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
        self.subplots = []

    
    def add_subplot(self, rci, **kwargs):
        ''' 
        rci -> rows, columns, index \n
        kwargs -> title, limits, projection, lables \n
        -------------------------------------------\n
        output -> ax\n
        '''
        title = kwargs.get('title', "")
        limits = kwargs.get('limits', ())
        projection = kwargs.get('projection', 'rectilinear')
        lables = kwargs.get('lables', [])
        '''rci -> rows, columns, index'''
        _ax = self.fig.add_subplot(rci, projection=projection)
        _params = {'ax': _ax, 'index': rci,
                   'limits': limits, 'title': title, 'projection': projection}
        if title:
            _ax.set_title(title)
            
        if projection == '3d':
            _ax.set_box_aspect([1, 1, 1])
        
        if lables:
            _ax.set_xlabel(lables[0])
            _ax.set_ylabel(lables[1])
            if projection == '3d':
                _ax.set_zlabel(lables[2])
        if limits:
            _ax.set_xlim(limits[0])
            _ax.set_ylim(limits[1])
            if projection == '3d':
                _ax.set_zlim(limits[2])

        self.subplots.append(_params)
        return _ax

    @classmethod
    def plot_path(cls, ax, path: np.ndarray) -> None:
        ax.plot(path[:, 0], path[:, 1], path[:, 2])

    @classmethod
    def plot_nodes(cls, ax, nodes: np.ndarray) -> None:
        ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], marker='o')

    
    @classmethod
    def plot_coordinate_frame(cls, ax, XYZwxyz: np.ndarray[7], size:float=0.1, linewidth:float=1) -> None:
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
        
    @classmethod
    def plot_geometric_path(cls, ax, geometric_path: np.ndarray[np.ndarray[7]], size:float=0.1, linewidth:float=1) -> None:
        for frame in geometric_path:
            cls.plot_coordinate_frame(ax, frame, size, linewidth)

    @classmethod
    def plot_motion_xyz(cls, ax: plt.Axes, geometric_path: np.ndarray[np.ndarray[7]]) -> None:
        ax.grid(True)
        ax.set_facecolor('lightgray')
        ax.set_title('Path')
        ax.plot(geometric_path[:, 0], color='blue')
        ax.plot(geometric_path[:, 1], color='red')
        ax.plot(geometric_path[:, 2], color='green')
        ax.legend(['X', 'Y', 'Z'])

    @classmethod
    def plot_motion_rpy(cls, ax: plt.Axes, geometric_path: np.ndarray[np.ndarray[7]]) -> None:
        ax.grid(True)
        ax.set_facecolor('lightgray')

        #apply rma.TxyzQwxyz_2_TxyzRxyz to get the euler angles to all rows
        euler_path = np.apply_along_axis(rma.TxyzQwxyz_2_TxyzRxyz, 1, geometric_path)
        euler_path[:, 3:] = rma.normalize_eulers(euler_path[:, 3:])

        ax.set_title('YPR')
        ax.plot(euler_path[:, 3], color='orange')
        ax.plot(euler_path[:, 4], color='magenta')
        ax.plot(euler_path[:, 5], color='brown')
        ax.legend(['Y', 'P', 'R'])

    @classmethod
    def plot_motion_quat(cls, ax: plt.Axes, geometric_path: np.ndarray[np.ndarray[7]]) -> None:
        ax.grid(True)
        ax.set_facecolor('lightgray')
        ax.set_title('Quaternion')
        ax.plot(geometric_path[:, 3], color='lightgreen')
        ax.plot(geometric_path[:, 4], color='orange')
        ax.plot(geometric_path[:, 5], color='blue')
        ax.plot(geometric_path[:, 6], color='black')
        ax.legend(['w', 'x', 'y', 'z'])

    
    @classmethod
    def plot_single_traj(cls, ax1, ax2, ax3, ax4, 
                         traj: np.ndarray[np.ndarray[7]], density:int=10) -> None:
        
        path = traj[:, 0:3]
        # nodes and frames are equally spaced points on the path
        nodes = path[::density]
        frames = traj[::density]
        cls.plot_path(ax1, path)
        cls.plot_nodes(ax1, nodes)
        cls.plot_geometric_path(ax1, frames, size=0.01, linewidth=1)
        cls.plot_motion_xyz(ax2, traj)
        cls.plot_motion_rpy(ax3, traj)
        cls.plot_motion_quat(ax4, traj)


    @classmethod
    def set_3D_plot_axis_limits(cls, ax, all_points) -> None:
        # Calculate the limits for the plot based on the geometric path
        minX = np.min(all_points[:, 0]); maxX = np.max(all_points[:, 0]); delta_X = maxX - minX
        minY = np.min(all_points[:, 1]); maxY = np.max(all_points[:, 1]); delta_Y = maxY - minY
        minZ = np.min(all_points[:, 2]); maxZ = np.max(all_points[:, 2]); delta_Z = maxZ - minZ
        max_range = max(delta_X, delta_Y, delta_Z)
        ax.set_xlim(minX, minX + max_range)
        ax.set_ylim(minY, minY + max_range)
        ax.set_zlim(minZ, minZ + max_range)

    
    def main(self, traj: np.ndarray[np.ndarray[7]], density:int=10) -> None:
        ax1 = self.add_subplot(121, projection='3d',
                               title='3D Spline Trajectory', 
                               lables=['X', 'Y', 'Z'])
        
        all_points = traj[:, 0:3]
        self.set_3D_plot_axis_limits(ax1, all_points)

        ax2 = self.add_subplot(333)
        ax3 = self.add_subplot(336)
        ax4 = self.add_subplot(339)

        self.set_3D_plot_axis_limits(ax1, traj)
        self.plot_single_traj(ax1, ax2, ax3, ax4, traj, density=density)
    
