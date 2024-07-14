from submodules.plot_traj_3d import PlotTraj3D
from submodules import robomath_addon as rma

import numpy as np

PlotTraj3D().main(traj=np.random.rand(100, 7), density=10)