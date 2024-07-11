import math
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
# from scipy.spatial.transform import Rotation
from matplotlib.backends.backend_pdf import PdfPages
import submodules.robomath as rm
import submodules.robomath_addon as rma
import submodules.motive_file_parser as mfp

# def euler_change_rishabh(Gwxyz) : #receives a list of quaternions
#     eulers = rma.quaternions_2_eulers(Gwxyz) #reshape the quaternions to a 2D array
    
#     rolls = eulers[:,0]
#     pitches = eulers[:,1]
#     yaws = eulers[:,2]

#     smooth_rolls = rma.normalize_eulers_method2(rolls)
#     smooth_pitches = rma.normalize_eulers_method2(pitches)
#     smooth_yaws = rma.normalize_eulers_method2(yaws)

#     smooth_eulers = np.array([smooth_rolls, smooth_pitches, smooth_yaws]).T

#     return smooth_eulers


# def euler_change(Gwxyz) : #receives a list of quaternions
#     quaternions = np.array(Gwxyz) #reshape the quaternions to a 2D array
#     eulers = rma.quaternions_2_eulers(Gwxyz)
#     smooth_eulers = np.apply_along_axis(rma.normalize_eulers, axis=0, arr=eulers)    
#     return smooth_eulers



print("..........")
path_name = "./../../datasets/dataset jun 28/Supporting Data - Sheet1.csv"
list = pd.read_csv(path_name)
base_path = "./../../datasets/dataset jun 28/sean & Li 2 exports/"
collums = list.columns
result_dict = {}
count = 0
for i in range(len(list)):
    if list[collums[3]][i] == 'accept':
        result_dict[count] = {
            'Path': base_path + str(list[collums[0]][i]) + '.csv',
            'start_frame': list[collums[1]][i],
            'end_frame': list[collums[2]][i],
            'Note': list[collums[4]][i]
        }
        count += 1


with PdfPages('experiment_results.pdf') as pdf:
    for key in result_dict:
        _data= extract_data(result_dict[key])
        # euler_Gwxyz = euler_change_old(Gwxyz)  # Convert quaternions to Euler angles
        # euler_Gwxyz = Gwxyz
        # {'GR', 'GP', 'SR', 'SP', 'BxR', 'BxP', 'BuR', 'BuP'}
        create_subplots(_data['GP'],
                        _data['GR'],
                        _data['SP'],
                        _data['SR'],
                        title=['Gripper Pos', 'Gripper Rot', 'Scooper Pos','Scooper Rot'],
                        layouts = [(2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4)],
                        sup_title = f"take_{key}")

        print(f"creating subplots for {key}")  
        pdf.savefig()
        plt.close()



# for key in result_dict:
#     _data= extract_data(result_dict[key])
#     print(_data)
#     # create_xy_video(_data['GP'], _data['BxP'], _data['SP'], fps=30, video_filename=f'videos/output_{key}.avi', marker_list=['o', 'x', '*'])
