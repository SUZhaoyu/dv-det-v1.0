import math

# ===========================================Dataset Configuration=============================================

aug_config = {'nbbox': 256,
              'rotate_range': 0,
              'rotate_mode': 'u',
              'scale_range': 0.,
              'scale_mode': 'u',
              'drop_out': 0.,
              'flip': False,
              'shuffle': False,
              'paste_augmentation': False,
              'paste_instance_num': 128,
              'maximum_interior_points': 100,
              'normalization': None}

bbox_padding = aug_config['nbbox']
diff_thres = 3
cls_thres = 0

# ===========================================Dimension Settings=============================================

dimension = [74, 84.0, 4.]
offset = [2., 42.0, 3.]
bev_resolution = [0.4, 0.4, 0.8]
padding_offset = 0.2

anchor_size = [1.6, 3.9, 1.5]
anchor_params = [[1.6, 3.9, 1.5, -1., 0.],
                 [1.6, 3.9, 1.5, -1., math.pi/2.]]

# ===========================================Model Hyper Parameters=============================================

grid_buffer_size = 3
output_pooling_size = 5
output_attr = 8
positive_thres = 0.6
negative_thres = 0.35

roi_thres = 0.3
iou_thres = 0.55
max_length = 256
roi_voxel_size = 5

# ===========================================Model Definition=============================================

base_params = {'base_00': {'subsample_res': 0.10, 'c_out':  16, 'kernel_res': 0.10, 'concat': True},
               'base_02': {'subsample_res': None, 'c_out':  16, 'kernel_res': 0.20, 'concat': False},
               'base_03': {'subsample_res': None, 'c_out':  16, 'kernel_res': None, 'concat': False},
               'base_04': {'subsample_res': 0.20, 'c_out':  16, 'kernel_res': None, 'concat': True},
               'base_05': {'subsample_res': None, 'c_out':  32, 'kernel_res': 0.40, 'concat': False},
               'base_06': {'subsample_res': None, 'c_out':  32, 'kernel_res': None, 'concat': False},
               'base_07': {'subsample_res': 0.40, 'c_out':  32, 'kernel_res': None, 'concat': True},
               'base_08': {'subsample_res': None, 'c_out':  64, 'kernel_res': [0.60, 0.60, 0.40], 'concat': False},
               'base_09': {'subsample_res': None, 'c_out':  64, 'kernel_res': None, 'concat': False},
               'base_10': {'subsample_res': None, 'c_out':  64, 'kernel_res': None, 'concat': True}}

bev_params = {'bev_00': {'kernel_size': 3, 'c_out': 128},
              'bev_01': {'kernel_size': 3, 'c_out': 128},
              'bev_02': {'kernel_size': 1, 'c_out': 2 * output_attr}}

refine_params = {'c_out': 128, 'kernel_size': 3, 'padding': 0.}
