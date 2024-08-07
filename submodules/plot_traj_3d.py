from matplotlib import pyplot as plt
import submodules.robomath as rm
import submodules.robomath_addon as rma
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

class PlotTraj3D(object):
    def __init__(self, fig_size=(15, 8)):
        print("3D_traj_plot.py is being run directly")
        self.fig = plt.figure(figsize=fig_size)
        self.fig.subplots_adjust(left=0.036, bottom=0.042, right=0.98, top=0.96, wspace=0.1, hspace=0.3)

        self.subplots = []
    
    def add_subplot(self, r, c, i, **kwargs):
        ''' 
        rci -> rows, columns, index \n
        kwargs -> title, limits, projection, lables \n
        -------------------------------------------\n
        output -> ax\n
        '''
        title = kwargs.get('title', "")
        suptitle = kwargs.get('suptitle', "")
        limits = kwargs.get('limits', ())
        projection = kwargs.get('projection', 'rectilinear')
        lables = kwargs.get('lables', [])
        '''rci -> rows, columns, index'''
        _ax = self.fig.add_subplot(r, c, i, projection=projection)
        _params = {'ax': _ax, 'index': (r, c, i),
                   'limits': limits, 'title': title, 'projection': projection}
        if title:
            _ax.set_title(title)

        if suptitle:
            plt.suptitle(suptitle)
            
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
        return ax.plot(path[:, 0], path[:, 1], path[:, 2])

    @classmethod
    def plot_nodes(cls, ax, nodes: np.ndarray, **kwargs) -> None:
        # Extract coordinates and labels
        labels = kwargs.get('labels', [])
        
        # Scatter plot
        scatter = ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], marker='o')

        # Add text labels
        for i, (label, (x, y, z)) in enumerate(zip(labels, nodes)):
            ax.text(x, y, z, label, fontsize=9, ha='right')

        return scatter
        
        # return ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], marker='o')
    
    # @classmethod
    # def plot_nodes(cls, ax, nodes: dict) -> None:
    #     """
    #     Plots nodes as a scatter plot and adds text labels at each node's position.

    #     Parameters:
    #     - ax: The matplotlib axis object to plot on.
    #     - nodes: A dictionary where keys are node labels and values are node coordinates (3D).
    #     """
    #     # Extract coordinates and labels
    #     labels = list(nodes.keys())
    #     coordinates = np.array(list(nodes.values()))
        
    #     # Scatter plot
    #     scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], marker='o')

    #     # Add text labels
    #     for i, (label, (x, y, z)) in enumerate(zip(labels, coordinates)):
    #         ax.text(x, y, z, label, fontsize=9, ha='right')

    #     return scatter
    
    @classmethod
    def plot_coordinate_frame(cls, ax, XYZwxyz: np.ndarray[7], size:float=0.1, linewidth:float=1) -> None:
        #frame is position + quaternion of size 7
        position = XYZwxyz[0:3]
        quaternion = XYZwxyz[3:]
        rotation_matrix = rm.quaternion_2_pose(quaternion)
        x_axis = rotation_matrix[:, 0] * size
        y_axis = rotation_matrix[:, 1] * size
        z_axis = rotation_matrix[:, 2] * size

        quivers = [
            ax.quiver(position[0], position[1], position[2], x_axis[0], x_axis[1], x_axis[2], color='r', linewidth=linewidth),
            ax.quiver(position[0], position[1], position[2], y_axis[0], y_axis[1], y_axis[2], color='g', linewidth=linewidth),
            ax.quiver(position[0], position[1], position[2], z_axis[0], z_axis[1], z_axis[2], color='b', linewidth=linewidth)
        ]
        return quivers
    

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
        if ax1:
            cls.plot_path(ax1, path)
            cls.plot_nodes(ax1, nodes)
            cls.plot_geometric_path(ax1, frames, size=0.03, linewidth=1)
        
        if ax2: cls.plot_motion_xyz(ax2, traj)
        if ax3: cls.plot_motion_rpy(ax3, traj)
        if ax4: cls.plot_motion_quat(ax4, traj)


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
    
    
    def animate_trajectory(self, ax, traj: np.ndarray[np.ndarray[7]], interval: int = 100) -> FuncAnimation:
        self.set_3D_plot_axis_limits(ax, traj[:, :3])

        line, = ax.plot([], [], [], 'r-', label='Trajectory')
        point, = ax.plot([], [], [], 'bo')
        quivers = []

        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            return [line, point] + quivers

        def update(num):
            line.set_data(traj[:num+1, 0], traj[:num+1, 1])
            line.set_3d_properties(traj[:num+1, 2])
            point.set_data(traj[num:num+1, 0], traj[num:num+1, 1])
            point.set_3d_properties(traj[num:num+1, 2])
            for q in quivers:
                q.remove()
            quivers[:] = self.plot_coordinate_frame(ax, traj[num], size=0.03, linewidth=2)
            return [line, point] + quivers

        ani = FuncAnimation(self.fig, update, frames=len(traj), init_func=init, blit=False, interval=interval)
        return ani

    def animate_multiple_trajectories(self, ax, list_trajectories: list[np.ndarray], interval: int = 100, **kwargs) -> FuncAnimation:
        """
            Animate multiple 3D trajectories.

            Parameters:
            list_trajectories (list[np.ndarray]): A list of 3D trajectories, each represented as a numpy array.
            interval (int, optional): The interval between frames in milliseconds. Default is 100.
            **kwargs:
            _qline_width =  kwargs.get('quiver_line_width', 1)
            _qsize = kwargs.get('quiver_size', 0.1)
            _pline_width = kwargs.get('path_line_width', 0.5)
            Returns:
            FuncAnimation: The animation object for the trajectories.
        """
        # ax = self.add_subplot(111, projection='3d', title='3D Spline Trajectories', labels=['X', 'Y', 'Z'])
        
        # Concatenate all trajectory points into a single array for setting plot limits
        # all_points = np.concatenate([traj[:, :3] for traj in list_trajectories], axis=0)
        # # Set axis limits based on the collected points
        # self.set_3D_plot_axis_limits(ax, all_points)

        # Initialize lines for each trajectory, empty at first, with a specific style and label
        _qline_width =  kwargs.get('quiver_line_width', 1)
        _qsize = kwargs.get('quiver_size', 0.1)
        _pline_width = kwargs.get('path_line_width', 0.5)

        lines = [ax.plot([], [], [], 'r-', linewidth=_pline_width, label=f'Trajectory {i}')[0] for i, _ in enumerate(list_trajectories)]

        # Initialize marker points for each trajectory, empty at first
        points = [ax.plot([], [], [], 'bo')[0] for _ in list_trajectories]

        # Create a list of lists for storing quivers associated with each trajectory
        quivers_lists = [[] for _ in list_trajectories]  # Each trajectory can have its own set of quivers

        def init():
            # Clear line and point data for resetting the plot
            for line, point in zip(lines, points):
                line.set_data([], [])
                line.set_3d_properties([])
                point.set_data([], [])
                point.set_3d_properties([])
            # Remove all quivers and clear their lists
            for quiver_list in quivers_lists:
                for q in quiver_list:
                    q.remove()
                quiver_list.clear()  # Clear the list after removing quivers

            # Return a list of all artists that need to be redrawn
            return lines + points + [item for sublist in quivers_lists for item in sublist]

        # Define an update function for the animation that gets called at each frame
        def update(num):
            # Update the positions of lines, points, and quivers for each trajectory
            for i, (line, point, quiver_list) in enumerate(zip(lines, points, quivers_lists)):
                # Get the current trajectory
                traj = list_trajectories[i]
                # Update the line and point positions
                # Set the x and y coordinates for 2D lines
                line.set_data(traj[:num + 1, 0], traj[:num + 1, 1])
                # Set the z-coordinate for 3D lines
                line.set_3d_properties(traj[:num + 1, 2])

                # Set the x, y, and z coordinates for 3D points
                point.set_data(traj[num:num + 1, 0], traj[num:num + 1, 1])
                
                # Set the z-coordinate for 3D points
                point.set_3d_properties(traj[num:num + 1, 2])

                # Correctly handle quivers
                # Remove old quivers properly
                # Remove existing quivers and clear the list
                while quiver_list:
                    q = quiver_list.pop()
                    # Remove the quiver from the plot
                    q.remove()

                # Add new quivers if orientation data is available
                if traj.shape[1] > 3:  # Check for orientation data
                    quiver_list.extend(self.plot_coordinate_frame(ax, traj[num], size=_qsize, linewidth=_qline_width))
            # Return a list of all artists that need to be redrawn
            return lines + points + [item for sublist in quivers_lists for item in sublist]

        ani = FuncAnimation(self.fig, update, frames=len(list_trajectories[0]), init_func=init, blit=False, interval=interval)
        return ani
    

