from submodules.plot_traj_3d import PlotTraj3D
from submodules import robomath_addon as rma

import numpy as np


#read csv file
import csv

file_path = './../datasets/Chisel_problem/30/cleaned_test_data/test_127_cleaned.csv'

with open(file_path, 'r', newline='') as file:
    reader = csv.reader(file)
    headers = ['gripper_x', 'gripper_y', 'gripper_z', 'gripper_w', 'gripper_X',
            'gripper_Y', 'gripper_Z', 'chisel_x', 'chisel_y', 'chisel_z', 'chisel_w', 'chisel_X', 'chisel_Y',
            'chisel_Z']
    


PlotTraj3D().main(traj=np.random.rand(100, 7), density=10)