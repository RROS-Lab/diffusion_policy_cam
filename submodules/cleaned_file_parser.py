import pandas as pd
# import submodules.robomath_addon as rma
import submodules.robomath_addon as rma
import numpy as np
# import submodules.data_filter as _df
import submodules.data_filter as _df
from typing import Union
import csv

def process_data(data :pd.DataFrame,
                 fps: float, filter: bool = False,
                 window_size: int = 15, polyorder: int = 3) -> pd.DataFrame:
    
    input_fps = float(data.iloc[0, 1])
    # print(input_fps)
    _new_data = _df.fps_sampler(data[1:], target_fps = fps, input_fps=input_fps)

    if filter:
        _new_data = _df.apply_savgol_filter(_new_data, window_size, polyorder, time_frame= True)
    return _new_data


class DataParser:
    """
        Cleaned DATA parser.
    """
    def __extract_column_info__(self):
        # Extract unique marker, rigid body, and tool names
        marker_columns = [col for col in self.data.columns if 'Marker' in col]
        self.markers = {value.split('_')[0] for value in self.data[marker_columns].iloc[0]}
        # self.markers = {val.split('_')[0] for column in self.data.columns if column.startswith('Marker') for val in self.data[column].iloc[0]}
        rigid_body_columns = [col for col in self.data.columns if 'RigidBody' in col]
        self.rigid_bodies = {value.split('_')[0] for value in self.data[rigid_body_columns].iloc[0]}
        # self.rigid_bodies = {val.split('_')[0] for column in self.data.columns if column.startswith('RigidBody') for val in self.data[column].iloc[0]}
        self.data.columns = self.data.iloc[0]
        self.data = self.data.drop(index =0)

    # @classmethod
    def get_rigid_TxyzQwxyz(self, **kwargs) -> dict[str: np.ndarray[np.ndarray[7]]]:
        """
        Process rigid body data from a DataFrame based on the specified type (QUAT or EULER).
        
        Args:
        - data (pd.DataFrame): The input DataFrame containing the data.
        - rigid_bodies (list): List of rigid body names to process.
        - data_type (Union['QUAT', 'EULER']): Type of data to process ('QUAT' for quaternions, 'EULER' for Euler angles).
        
        Returns:
        - dict: Dictionary containing processed rigid body data for each rigid body.
        """
        rb_TxyzQwxyz = {}  # Dictionary to store processed data
        
        for rb in self.rigid_bodies:
            rb_columns = [col for col in self.data.columns if col.startswith(rb)]
            sorted_columns = sorted(rb_columns, key=lambda x: x.split('_')[1])
            if rb+'_state' in sorted_columns:
                sorted_columns.remove(rb+'_state')
            # Select processing method based on data type and column length
            if self.file_type == 'QUAT':
                #motive data
                # rb_TxyzQwxyz[rb] = np.apply_along_axis(rma.motive_2_robodk_rigidbody, 1, self.data[sorted_columns].values.astype(float))
                rb_TxyzQwxyz[rb] = self.data[sorted_columns].values.astype(float)

            elif self.file_type == 'EULER':
                rb_TxyzQwxyz[rb] = np.apply_along_axis(rma.TxyzRxyz_2_TxyzQwxyz, 1, self.data[sorted_columns].values.astype(float))

        for key, value in kwargs.items():
            if key == 'item':
                return {key: rb_TxyzQwxyz[key] for key in value if key in rb_TxyzQwxyz}
        
        return rb_TxyzQwxyz
    
    def get_rigid_TxyzRxyz(self, **kwargs) -> dict[str: np.ndarray[np.ndarray[6]]]:
        """
        Process rigid body data from a DataFrame based on the specified type (QUAT or EULER).
        
        Args:
        - data (pd.DataFrame): The input DataFrame containing the data.
        - rigid_bodies (list): List of rigid body names to process.
        - data_type (Union['QUAT', 'EULER']): Type of data to process ('QUAT' for quaternions, 'EULER' for Euler angles).
        
        Returns:
        - dict: Dictionary containing processed rigid body data for each rigid body.
        """
        rb_TxyzRxyz = {}  # Dictionary to store processed data
        
        for rb in self.rigid_bodies:
            rb_columns = [col for col in self.data.columns if col.startswith(rb)]
            sorted_columns = sorted(rb_columns, key=lambda x: x.split('_')[1]) 
            if rb+'_state' in sorted_columns:
                sorted_columns.remove(rb+'_state')
            
            # Select processing method based on data type and column length
            if self.file_type == 'QUAT':
                rb_TxyzRxyz[rb] = np.apply_along_axis(rma.TxyzQwxyz_2_TxyzRxyz, 1, self.data[sorted_columns].values.astype(float))
            elif self.file_type == 'EULER':
                rb_TxyzRxyz[rb] = self.data[rb_columns].values.astype(float)

        for key, value in kwargs.items():
            if key == 'item':
                return {key: rb_TxyzRxyz[key] for key in value if key in rb_TxyzRxyz}
    
        return rb_TxyzRxyz


    # @classmethod
    def get_marker_Txyz(self, **kwargs) -> dict[str: np.ndarray[np.ndarray[3]]]:
        """
        Process marker data from a DataFrame.
        
        Args:
        - data (pd.DataFrame): The input DataFrame containing the data.
        - markers (list): List of marker names to process.
        
        Returns:
        - dict: Dictionary containing processed marker data for each marker.
        """
        mk_Txyz = {}

        # Extract marker data
        for mk in self.markers:
            mk_columns = [col for col in self.data.columns if col.startswith(mk)]
            sorted_columns = sorted(mk_columns, key=lambda x: x.split('_')[1])
            
            # Motive
            # mk_Txyz[mk] = np.apply_along_axis(rma.motive_2_robodk_marker, 1, self.data[mk_columns].values.astype(float))
            mk_Txyz[mk] = self.data[sorted_columns].values.astype(float)

        for key, value in kwargs.items():
            if key == 'object':
                return {key: mk_Txyz[key] for key in value if key in mk_Txyz}

        return mk_Txyz

    def get_rigid_state(self, **kwargs) -> dict[str: np.ndarray]:
        """
        get state of rigid body data from a DataFrame.
        
        Args:
        - data (pd.DataFrame): The input DataFrame containing the data.
        - rigid_bodies (list): List of rigid body names to process.
        
        Returns:
        - dict: Dictionary containing processed rigid body state data for each rigid body.
        """
        rb_state = {}  # Dictionary to store processed data
        
        for rb in self.rigid_bodies:
            rb_columns = [col for col in self.data.columns if col.startswith(rb) and col.endswith('state')]
            # Select processing method based on data type and column length
            rb_state[rb] = self.data[rb_columns].values

        for key, value in kwargs.items():
            if key == 'item':
                return {key: rb_state[key] for key in value if key in rb_state}
        
        return rb_state
    

    def get_time(self, **kwargs) -> np.ndarray:
        """
        Get time data from a DataFrame.
        
        Args:
        - data (pd.DataFrame): The input DataFrame containing the data.
        
        Returns:
        - np.ndarray: Array containing time data.
        """
        time = {}  # Dictionary to store processed data
        time = self.data['Time'].values
        time = time.astype(float)
        return time

    def save_2_csv(self, file_path, save_type='QUAT'):
        # add first rows
        _params = {
            'QUAT': {'len':7,
                     'dof': ['X', 'Y', 'Z', 'w', 'x', 'y', 'z'],
                     '__gettr__': self.get_rigid_TxyzQwxyz },
            'EULER': {'len':6,
                      'dof': ['X', 'Y', 'Z', 'x', 'y', 'z'],
                      '__gettr__': self.get_rigid_TxyzRxyz}
        }
        
        _SUP_HEADER_ROW = (['Time_stamp']+["RigidBody"] * len(self.rigid_bodies) * _params[save_type]['len'] + ["Marker"] * len(self.markers) * 3)
        _FPS_ROW = ["FPS", self.fps] + [0.0]*(len(_SUP_HEADER_ROW) - 2)
        _rb_col_names = [f"{rb}_{axis}" for rb in self.rigid_bodies for axis in _params[save_type]['dof']]
        _mk_col_names = [f"{mk}_{axis}" for mk in self.markers for axis in ['X', 'Y', 'Z']]
        _HEADER_ROW = ['Time']+_rb_col_names + _mk_col_names

        _dict_data_rigid = _params[save_type]['__gettr__']()
        _dict_data_marker = self.get_marker_Txyz()
        _dict_data_time = self.get_time()
        _dict_data_time = _dict_data_time.reshape((len(_dict_data_time), 1))

        # concatenate all the data into a single array for _dict_data_rigid
        _transformed_data_rigid = np.concatenate([_dict_data_rigid[rb] for rb in self.rigid_bodies], axis=1)
        _transformed_data_marker = np.concatenate([_dict_data_marker[mk] for mk in self.markers], axis=1)
        _transformed_data = np.concatenate([_dict_data_time, _transformed_data_rigid, _transformed_data_marker], axis=1)

        # save as csv file with SUP_HEADER_ROW, FPS_ROW, HEADER_ROW, and _transformed_data
        with open(file_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(_SUP_HEADER_ROW)
            writer.writerow(_FPS_ROW)
            writer.writerow(_HEADER_ROW)
            writer.writerows(_transformed_data)



    @classmethod
    def from_euler_file(self, file_path, target_fps: Union[float , None], filter: bool = False, window_size: int = 15, polyorder: int = 3):

        return DataParser(file_path, 'EULER', target_fps, filter, window_size, polyorder)
    

    
    @classmethod
    def from_quat_file(self, file_path, target_fps: Union[float , None], filter: bool = False, window_size: int = 15, polyorder: int = 3):
        return DataParser(file_path, 'QUAT', target_fps, filter, window_size, polyorder)
    

    def __init__(self, file_path, file_type: Union['QUAT', 'EULER'], target_fps: float, 
                 filter: bool = False, window_size: int = 15, polyorder: int = 3):
        
        self.data = process_data(pd.read_csv(file_path), target_fps, filter, window_size, polyorder)
        self.fps = float(target_fps)
        self.markers = set()
        self.rigid_bodies = set()
        self.file_type = file_type
        self.__extract_column_info__()


# if __name__ == "__main__":

#     path = '/home/cam/Documents/diffusion_policy_cam/diffusion_pipline/data_chisel_task/cap_008_cleaned.csv'

#     data = DataParser.from_quat_file(file_path = path, target_fps=30.0, filter=False, window_size=15, polyorder=3)

#     print(data.rigid_bodies)

    # tools = data.get_rigid_TxyzQwxyz(object = ['chisel'])

#     print(tools)