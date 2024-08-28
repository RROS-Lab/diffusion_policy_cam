import pandas as pd
import numpy as np
from typing import Union
import csv



class ForceParser:
    """
        ATI DATA parser.
    """

    # @classmethod
    def get_force_data(self, **kwargs) -> np.ndarray[np.ndarray[6]]:
        """
        Get force data from a DataFrame. If specific indices are provided,
        return data only for those indices.
        """
        # Extract indices from kwargs, default to None if not provided
        indices = kwargs.get('indices', None)

        # If indices are provided, filter the DataFrame rows based on those indices
        if indices is not None:
            # Ensure indices is a list or a single index
            if not isinstance(indices, (list, tuple)):
                indices = [indices]
            filtered_data = self.data.iloc[indices, 1:]
        else:
            # If no indices are provided, use all rows
            filtered_data = self.data.iloc[:, 1:]
        
        # Convert filtered data to a NumPy array
        force_data = filtered_data.to_numpy()
        
        return force_data
    

    def get_time(self, **kwargs) -> np.ndarray:
        """
        Get time data from a DataFrame.
        
        Args:
        - data (pd.DataFrame): The input DataFrame containing the data.
        
        Returns:
        - np.ndarray: Array containing time data.
        """
        time = self.data.iloc[:, :1].to_numpy().flatten()
        return time



    @classmethod
    def from_euler_file(self, file_path, **kwargs):
        return ForceParser(file_path, 'EULER')
    


    def __init__(self, file_path, file_type: Union["QUAT", "EULER"]):
        
        self.data = pd.read_csv(file_path)
        self.file_type = file_type



# if __name__ == "__main__":

#     path = '/home/cam/Documents/raj/diffusion_policy_cam/no-sync/turn_table_chisel/dataset_aug14/ft_data_200/takes/ft_002.csv'

#     data = ForceParser.from_quat_file(file_path = path)

#     print(data.get_force_data().shape)