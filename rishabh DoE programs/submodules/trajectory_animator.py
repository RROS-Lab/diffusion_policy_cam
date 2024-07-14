import numpy as np
from matplotlib.animation import FuncAnimation

class TrajectoryAnimator:
    def __init__(self, fig):
        self.fig = fig

    def animate_multiple_trajectories(self, trajectories, interval=100):
        ax = self._setup_plot()
        lines, points, quivers = self._initialize_plot_elements(ax, trajectories)
        
        def init():
            return self._init_animation(ax, lines, points, quivers)

        def update(frame):
            return self._update_frame(ax, lines, points, quivers, trajectories, frame)

        ani = FuncAnimation(self.fig, update, frames=len(trajectories[0]), init_func=init, blit=False, interval=interval)
        return ani

    def _setup_plot(self):
        ax = self.fig.add_subplot(111, projection='3d')
        ax.set_title('3D Spline Trajectories')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        return ax

    def _initialize_plot_elements(self, ax, trajectories):
        all_points = np.concatenate([traj[:, :3] for traj in trajectories], axis=0)
        self._set_3d_plot_axis_limits(ax, all_points)

        lines = [ax.plot([], [], [], 'r-', label=f'Trajectory {i}')[0] for i in range(len(trajectories))]
        points = [ax.plot([], [], [], 'bo')[0] for _ in trajectories]
        quivers = [[] for _ in trajectories]
        return lines, points, quivers

    def _init_animation(self, ax, lines, points, quivers):
        for line, point in zip(lines, points):
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
        for quiver_list in quivers:
            self._remove_quivers(quiver_list)
        return lines + points + [item for sublist in quivers for item in sublist]

    def _update_frame(self, ax, lines, points, quivers, trajectories, frame):
        for i, (line, point, quiver_list) in enumerate(zip(lines, points, quivers)):
            traj = trajectories[i]
            line.set_data(traj[:frame + 1, 0], traj[:frame + 1, 1])
            line.set_3d_properties(traj[:frame + 1, 2])
            point.set_data(traj[frame:frame + 1, 0], traj[frame:frame + 1, 1])
            point.set_3d_properties(traj[frame:frame + 1, 2])
            self._update_quivers(ax, quiver_list, traj, frame)
        return lines + points + [item for sublist in quivers for item in sublist]

    def _update_quivers(self, ax, quiver_list, trajectory, frame):
        self._remove_quivers(quiver_list)
        if trajectory.shape[1] > 3:  # Assuming extra columns for orientations
            quiver_list.extend(self._plot_coordinate_frame(ax, trajectory[frame], size=0.1, linewidth=1))

    def _remove_quivers(self, quiver_list):
        while quiver_list:
            quiver_list.pop().remove()

    def _set_3d_plot_axis_limits(self, ax, points):
        ax.set_xlim(points[:, 0].min(), points[:, 0].max())
        ax.set_ylim(points[:, 1].min(), points[:, 1].max())
        ax.set_zlim(points[:, 2].min(), points[:, 2].max())

    def _plot_coordinate_frame(self, ax, point, size=0.1, linewidth=1):
        # Example implementation to plot coordinate frames
        return [ax.quiver(point[0], point[1], point[2], size, 0, 0, color='r', linewidth=linewidth),
                ax.quiver(point[0], point[1], point[2], 0, size, 0, color='g', linewidth=linewidth),
                ax.quiver(point[0], point[1], point[2], 0, 0, size, color='b', linewidth=linewidth)]
