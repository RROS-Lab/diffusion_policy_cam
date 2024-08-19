from scipy.interpolate import interp1d
from scipy.interpolate import griddata
import numpy as np

def temporal_interpolation(markers_dict, time_stamps):
    for marker, positions in markers_dict.items():
        for i in range(positions.shape[1]):  # Iterate over x, y, z
            valid_mask = ~np.isnan(positions[:, i])
            if np.sum(valid_mask) < 2:
                # Not enough points to interpolate
                continue
            interp_func = interp1d(time_stamps[valid_mask], positions[valid_mask, i], kind='linear', fill_value="extrapolate")
            positions[:, i] = interp_func(time_stamps)
    return markers_dict

def spatial_interpolation(markers_dict):
    time_steps = markers_dict[list(markers_dict.keys())[0]].shape[0]
    
    for t in range(time_steps):
        positions = []
        grid_coords = []
        
        for marker, pos in markers_dict.items():
            if not np.isnan(pos[t]).any():
                positions.append(pos[t])
                row = ord(marker[0]) - ord('A')
                col = int(marker[1]) - 1
                grid_coords.append([row, col])
        
        grid_coords = np.array(grid_coords)
        positions = np.array(positions)
        
        grid_x, grid_y = np.mgrid[0:5, 0:5]
        
        interpolated_x = griddata(grid_coords, positions[:, 0], (grid_x, grid_y), method='cubic')
        interpolated_y = griddata(grid_coords, positions[:, 1], (grid_x, grid_y), method='cubic')
        interpolated_z = griddata(grid_coords, positions[:, 2], (grid_x, grid_y), method='cubic')
        
        for marker in markers_dict.keys():
            row = ord(marker[0]) - ord('A')
            col = int(marker[1]) - 1
            if np.isnan(markers_dict[marker][t]).any():
                markers_dict[marker][t, 0] = interpolated_x[row, col]
                markers_dict[marker][t, 1] = interpolated_y[row, col]
                markers_dict[marker][t, 2] = interpolated_z[row, col]
    
    return markers_dict

def interpolate_markers(markers_dict, time_stamps):
    markers_dict = temporal_interpolation(markers_dict, time_stamps)
    markers_dict = spatial_interpolation(markers_dict)
    return markers_dict

