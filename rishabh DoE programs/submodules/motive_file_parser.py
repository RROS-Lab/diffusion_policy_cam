import pandas as pd

def fps_sampler(data: pd.DataFrame, target_fps:float, input_fps: float = 240.0):
    sample_size = int(input_fps / target_fps)
    _output_data = []
    #get every nth row of a dataframe
    _output_data = data.iloc[::sample_size]
    return _output_data


def extract_data_chisel(file_path: str):
    CXYZ, Cwxyz, GXYZ, Gwxyz, BXYZ, Bwxyz, A1XYZ, A2XYZ, A3XYZ, B1XYZ, B2XYZ, B3XYZ, C1XYZ, C2XYZ, C3XYZ = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    desired_order = ['chisel_x', 'chisel_y', 'chisel_z', 'chisel_w', 'chisel_X', 'chisel_Y',
            'chisel_Z', 'gripper_x', 'gripper_y', 'gripper_z', 'gripper_w', 'gripper_X',
            'gripper_Y', 'gripper_Z', 'battery_x', 'battery_y', 'battery_z', 'battery_w',
            'battery_X', 'battery_Y', 'battery_Z', 'A1_X', 'A1_Y', 'A1_Z', 'A2_X', 'A2_Y',
            'A2_Z', 'A3_X', 'A3_Y', 'A3_Z', 'B1_X', 'B1_Y', 'B1_Z', 'B2_X', 'B2_Y', 'B2_Z',
            'B3_X', 'B3_Y', 'B3_Z', 'C1_X', 'C1_Y', 'C1_Z', 'C2_X', 'C2_Y', 'C2_Z',
            'C3_X', 'C3_Y', 'C3_Z']


    data = pd.read_csv(file_path)
    dict_of_lists = data.to_dict('list')
    chisel_pos = [tuple(item) for item in zip(dict_of_lists[desired_order[0]], dict_of_lists[desired_order[1]], dict_of_lists[desired_order[2]])]
    # chisel_pos = sampler(chisel_pos, sample_size)
    chisel_rot = [tuple(item) for item in zip(dict_of_lists[desired_order[3]], dict_of_lists[desired_order[4]], dict_of_lists[desired_order[5]], dict_of_lists[desired_order[6]])]
    # chisel_rot = sampler(chisel_rot, sample_size)
    gripper_pos = [tuple(item) for item in zip(dict_of_lists[desired_order[7]], dict_of_lists[desired_order[8]], dict_of_lists[desired_order[9]])]
    # gripper_pos = sampler(gripper_pos, sample_size)
    gripper_rot = [tuple(item) for item in zip(dict_of_lists[desired_order[10]], dict_of_lists[desired_order[11]], dict_of_lists[desired_order[12]], dict_of_lists[desired_order[13]])]
    # gripper_rot = sampler(gripper_rot, sample_size)
    battery_pos = [tuple(item) for item in zip(dict_of_lists[desired_order[14]], dict_of_lists[desired_order[15]], dict_of_lists[desired_order[16]])]
    # battery_pos = sampler(battery_pos, sample_size)
    battery_rot = [tuple(item) for item in zip(dict_of_lists[desired_order[17]], dict_of_lists[desired_order[18]], dict_of_lists[desired_order[19]], dict_of_lists[desired_order[20]])]
    # battery_rot = sampler(battery_rot, sample_size)
    A1_pos = [tuple(item) for item in zip(dict_of_lists[desired_order[21]], dict_of_lists[desired_order[22]], dict_of_lists[desired_order[23]])]
    # A1_pos = sampler(A1_pos, sample_size)
    A2_pos = [tuple(item) for item in zip(dict_of_lists[desired_order[24]], dict_of_lists[desired_order[25]], dict_of_lists[desired_order[26]])]
    # A2_pos = sampler(A2_pos, sample_size)
    A3_pos = [tuple(item) for item in zip(dict_of_lists[desired_order[27]], dict_of_lists[desired_order[28]], dict_of_lists[desired_order[29]])]
    # A3_pos = sampler(A3_pos, sample_size)
    B1_pos = [tuple(item) for item in zip(dict_of_lists[desired_order[30]], dict_of_lists[desired_order[31]], dict_of_lists[desired_order[32]])]
    # B1_pos = sampler(B1_pos, sample_size)
    B2_pos = [tuple(item) for item in zip(dict_of_lists[desired_order[33]], dict_of_lists[desired_order[34]], dict_of_lists[desired_order[35]])]
    # B2_pos = sampler(B2_pos, sample_size)
    B3_pos = [tuple(item) for item in zip(dict_of_lists[desired_order[36]], dict_of_lists[desired_order[37]], dict_of_lists[desired_order[38]])]
    # B3_pos = sampler(B3_pos, sample_size)
    C1_pos = [tuple(item) for item in zip(dict_of_lists[desired_order[39]], dict_of_lists[desired_order[40]], dict_of_lists[desired_order[41]])]
    # C1_pos = sampler(C1_pos, sample_size)
    C2_pos = [tuple(item) for item in zip(dict_of_lists[desired_order[42]], dict_of_lists[desired_order[43]], dict_of_lists[desired_order[44]])]
    # C2_pos = sampler(C2_pos, sample_size)
    C3_pos = [tuple(item) for item in zip(dict_of_lists[desired_order[45]], dict_of_lists[desired_order[46]], dict_of_lists[desired_order[47]])]
    # C3_pos = sampler(C3_pos, sample_size)

    # Now to modify each sublist in chisel_pos
    chisel_pos = [(sublist[2], sublist[0], sublist[1]) for sublist in chisel_pos]
    gripper_pos = [(sublist[2], sublist[0], sublist[1]) for sublist in gripper_pos]
    battery_pos = [(sublist[2], sublist[0], sublist[1]) for sublist in battery_pos]
    A1_pos = [(sublist[2], sublist[0], sublist[1]) for sublist in A1_pos]
    A2_pos = [(sublist[2], sublist[0], sublist[1]) for sublist in A2_pos]
    A3_pos = [(sublist[2], sublist[0], sublist[1]) for sublist in A3_pos]
    B1_pos = [(sublist[2], sublist[0], sublist[1]) for sublist in B1_pos]
    B2_pos = [(sublist[2], sublist[0], sublist[1]) for sublist in B2_pos]
    B3_pos = [(sublist[2], sublist[0], sublist[1]) for sublist in B3_pos]
    C1_pos = [(sublist[2], sublist[0], sublist[1]) for sublist in C1_pos]
    C2_pos = [(sublist[2], sublist[0], sublist[1]) for sublist in C2_pos]
    C3_pos = [(sublist[2], sublist[0], sublist[1]) for sublist in C3_pos]

    # For rotation, if you have w, x, y, z and want w, x, y, z re-ordered to w, x, y, z:
    chisel_rot = [(sublist[3], sublist[2], sublist[0], sublist[1]) for sublist in chisel_rot]
    gripper_rot = [(sublist[3], sublist[2], sublist[0], sublist[1]) for sublist in gripper_rot]
    battery_rot = [(sublist[3], sublist[2], sublist[0], sublist[1]) for sublist in battery_rot]


    CXYZ.extend(chisel_pos)
    Cwxyz.extend(chisel_rot)
    GXYZ.extend(gripper_pos)
    Gwxyz.extend(gripper_rot)
    BXYZ.extend(battery_pos)
    Bwxyz.extend(battery_rot)
    A1XYZ.extend(A1_pos)
    A2XYZ.extend(A2_pos)
    A3XYZ.extend(A3_pos)
    B1XYZ.extend(B1_pos)
    B2XYZ.extend(B2_pos)
    B3XYZ.extend(B3_pos)
    C1XYZ.extend(C1_pos)
    C2XYZ.extend(C2_pos)
    C3XYZ.extend(C3_pos)

    return CXYZ, Cwxyz, GXYZ, Gwxyz, BXYZ, Bwxyz, A1XYZ, A2XYZ, A3XYZ, B1XYZ, B2XYZ, B3XYZ, C1XYZ, C2XYZ, C3XYZ

