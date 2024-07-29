import pandas as pd
# import submodules.robomath_addon as rma
import submodules.robomath_addon as rma
import numpy as np
# import submodules.data_filter as _df
import submodules.data_filter as _df
from typing import Union
import csv

def process_data(data: pd.DataFrame,
                 output_fps: float, input_fps: float,
                 filter: bool = False,
                 window_size: int = 15, polyorder: int = 3) -> pd.DataFrame:
    """
    Process data by resampling and optionally applying a filter.

    Args:
    - data (pd.DataFrame): Input data.
    - output_fps (float): Desired output frames per second.
    - input_fps (float): Input frames per second.
    - filter (bool): Whether to apply a filter.
    - window_size (int): Window size for the filter.
    - polyorder (int): Polynomial order for the filter.

    Returns:
    - pd.DataFrame: Processed data.
    """
    time_col = data['Time_stamp'].copy()
    new_data = _df.fps_sampler(data.drop(columns=['Time_stamp']), target_fps=output_fps, input_fps=input_fps)

    if filter:
        new_data = _df.apply_savgol_filter(new_data, window_size, polyorder)

    new_data.insert(0, 'Time', time_col.values[:new_data.shape[0]])  # Reinsert time column
    return new_data

class DataParser:
    """
        Cleaned DATA parser.
    """
    def __init__(self, data, file_type: Union['QUAT', 'EULER'], target_fps: float, 
                 filter: bool = False, window_size: int = 15, polyorder: int = 3):
        self.data = data
        self.fps = float(target_fps)
        self.markers = set()
        self.rigid_bodies = set()
        self.file_type = file_type
        self.__extract_column_info__()

    def __extract_column_info__(self):
        """
        Extract unique marker and rigid body names from the data.
        """
        # Set the column headers correctly
        self.data.columns = self.data.iloc[0]
        self.data = self.data.drop(index=0).reset_index(drop=True)

        # Extract marker and rigid body names
        marker_columns = [col for col in self.data.columns if 'Marker' in col]
        rigid_body_columns = [col for col in self.data.columns if 'RigidBody' in col]
        
        self.markers = {col.split('_')[0] for col in marker_columns}
        self.rigid_bodies = {col.split('_')[0] for col in rigid_body_columns}
        
        print("Extracted markers:", self.markers)
        print("Extracted rigid bodies:", self.rigid_bodies)


    @classmethod
    def _from_file(cls, file_path: str, target_fps: Union[float, None], file_type: str, filter: bool = False, window_size: int = 15, polyorder: int = 3) -> 'DataParser':
        csv_data = pd.read_csv(file_path)
        input_fps = float(csv_data.iloc[0, 1])
        new_data = process_data(csv_data[1:], target_fps, input_fps, filter, window_size, polyorder)
        return cls(new_data, file_type, target_fps, filter, window_size, polyorder)

    @classmethod
    def from_euler_file(cls, file_path: str, target_fps: Union[float, None], filter: bool = False, window_size: int = 15, polyorder: int = 3) -> 'DataParser':
        return cls._from_file(file_path, target_fps, 'EULER', filter, window_size, polyorder)

    @classmethod
    def from_quat_file(cls, file_path: str, target_fps: Union[float, None], filter: bool = False, window_size: int = 15, polyorder: int = 3) -> 'DataParser':
        return cls._from_file(file_path, target_fps, 'QUAT', filter, window_size, polyorder)


    # @classmethod
    # def from_dict(cls, data_dict: dict[str: np.ndarray]) -> 'DataParser':
    #     '''
    #     data -> 'rb' : {'rb1': np.ndarray[np.ndarray[]], 'rb2': np.ndarray[np.ndarray[]]}
    #          -> 'mk' : {'mk1': np.ndarray[np.ndarray[3]], 'mk2': np.ndarray[np.ndarray[3]]}
    #          -> 'time' : np.ndarray
    #          -> 'type' : 'QUAT' or 'EULER'
    #          -> 'fps' : float
    #     '''
    #     #create a dataframe from data_dict
    #     _data = pd.DataFrame()
    #     _type = data_dict['type']
    #     _fps = data_dict['fps']
    #     _rb_names = data_dict['rb'].keys()
    #     _mk_names = data_dict['mk'].keys()
        


    #     return cls(_data, 'QUAT', None, False, 15, 3)

    # @classmethod
    def get_rigid_TxyzQwxyz(self, **kwargs) -> dict[str, np.ndarray]:
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
            if rb + '_state' in sorted_columns:
                sorted_columns.remove(rb + '_state')
            
            # Debug print
            print(f"Processing rigid body '{rb}' with columns: {sorted_columns}")
            
            # Select processing method based on data type and column length
            if self.file_type == 'QUAT':
                rb_TxyzQwxyz[rb] = self.data[sorted_columns].values.astype(float)
            elif self.file_type == 'EULER':
                rb_TxyzQwxyz[rb] = np.apply_along_axis(rma.TxyzRxyz_2_TxyzQwxyz, 1, self.data[sorted_columns].values.astype(float))
        
        for key, value in kwargs.items():
            if key == 'item':
                return {k: rb_TxyzQwxyz[k] for k in value if k in rb_TxyzQwxyz}
        
        return rb_TxyzQwxyz

    
    def get_rigid_TxyzRxyz(self, **kwargs) -> dict[str, np.ndarray]:
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
            if rb + '_state' in sorted_columns:
                sorted_columns.remove(rb + '_state')
            
            # Debug print
            print(f"Processing rigid body '{rb}' with columns: {sorted_columns}")
            
            # Select processing method based on data type and column length
            if self.file_type == 'QUAT':
                rb_TxyzRxyz[rb] = np.apply_along_axis(rma.TxyzQwxyz_2_TxyzRxyz, 1, self.data[sorted_columns].values.astype(float))
            elif self.file_type == 'EULER':
                rb_TxyzRxyz[rb] = self.data[sorted_columns].values.astype(float)
        
        for key, value in kwargs.items():
            if key == 'item':
                return {k: rb_TxyzRxyz[k] for k in value if k in rb_TxyzRxyz}
        
        return rb_TxyzRxyz


    def get_marker_Txyz(self, **kwargs) -> dict[str, np.ndarray]:
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
            
            # Debug print
            print(f"Processing marker '{mk}' with columns: {sorted_columns}")
            
            mk_Txyz[mk] = self.data[sorted_columns].values.astype(float)

        for key, value in kwargs.items():
            if key == 'marker':
                return {k: mk_Txyz[k] for k in value if k in mk_Txyz}

        return mk_Txyz


    def get_rigid_state(self, **kwargs) -> dict[str, np.ndarray]:
        """
        Get state of rigid body data from a DataFrame.
        
        Args:
        - data (pd.DataFrame): The input DataFrame containing the data.
        - rigid_bodies (list): List of rigid body names to process.
        
        Returns:
        - dict: Dictionary containing processed rigid body state data for each rigid body.
        """
        rb_state = {}  # Dictionary to store processed data
        
        for rb in self.rigid_bodies:
            rb_columns = [col for col in self.data.columns if col.startswith(rb) and col.endswith('state')]
            
            # Debug print
            print(f"Processing rigid body state '{rb}' with columns: {rb_columns}")
            
            rb_state[rb] = self.data[rb_columns].values

        for key, value in kwargs.items():
            if key == 'item':
                return {k: rb_state[k] for k in value if k in rb_state}
        
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
        
        _SUP_HEADER_ROW = (["RigidBody"] * len(self.rigid_bodies) * _params[save_type]['len'] + ["Marker"] * len(self.markers) * 3)
        _FPS_ROW = ["FPS", self.fps] + [0.0]*(len(_SUP_HEADER_ROW) - 2)
        _rb_col_names = [f"{rb}_{axis}" for rb in self.rigid_bodies for axis in _params[save_type]['dof']]
        _mk_col_names = [f"{mk}_{axis}" for mk in self.markers for axis in ['X', 'Y', 'Z']]
        _HEADER_ROW = _rb_col_names + _mk_col_names

        _dict_data_rigid = _params[save_type]['__gettr__']()
        _dict_data_marker = self.get_marker_Txyz()

        # concatenate all the data into a single array for _dict_data_rigid
        _transformed_data_rigid = np.concatenate([_dict_data_rigid[rb] for rb in self.rigid_bodies], axis=1)
        _transformed_data_marker = np.concatenate([_dict_data_marker[mk] for mk in self.markers], axis=1)
        _transformed_data = np.concatenate([_transformed_data_rigid, _transformed_data_marker], axis=1)

        # save as csv file with SUP_HEADER_ROW, FPS_ROW, HEADER_ROW, and _transformed_data
        with open(file_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(_SUP_HEADER_ROW)
            writer.writerow(_FPS_ROW)
            writer.writerow(_HEADER_ROW)
            writer.writerows(_transformed_data)



    
    

    


# if __name__ == "__main__":

#     path = '/home/cam/Documents/diffusion_policy_cam/diffusion_pipline/data_chisel_task/cap_008_cleaned.csv'

#     data = DataParser.from_quat_file(file_path = path, target_fps=30.0, filter=False, window_size=15, polyorder=3)

#     print(data.rigid_bodies)

#     tools = data.get_rigid_TxyzQwxyz(item = ['chisel'])

#     print(tools)