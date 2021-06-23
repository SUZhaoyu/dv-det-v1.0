import os
import math

# ===========================================Dataset Configuration=============================================

aug_config = {'nbbox': 256,
              'rotate_range': math.pi / 4,
              'rotate_mode': 'u',
              'scale_range': 0.05,
              'scale_mode': 'u',
              'drop_out': 0.1,
              'flip': True,
              'shuffle': True,
              'paste_augmentation': True,
              'paste_instance_num': 128,
              'maximum_interior_points': 100,
              'normalization': None}

bbox_padding = aug_config['nbbox']
diff_thres = 3
cls_thres = 0

# ===========================================Dimension Settings=============================================

dimension = [100., 160., 4.]
offset = [10., 80., 3.]
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
forge_ignore_thres = 0.1

roi_thres = 0.3
iou_thres = 0.55
roi_voxel_size = 5

# ===========================================Model Definition=============================================

base_params = {'base_00': {'subsample_res': 0.10, 'c_out':  16, 'kernel_res': 0.10, 'concat': True},
               'base_02': {'subsample_res': None, 'c_out':  16, 'kernel_res': 0.20, 'concat': False},
               'base_03': {'subsample_res': None, 'c_out':  16, 'kernel_res': 0.20, 'concat': False},
               'base_04': {'subsample_res': 0.20, 'c_out':  16, 'kernel_res': 0.20, 'concat': True},
               'base_05': {'subsample_res': None, 'c_out':  32, 'kernel_res': 0.40, 'concat': False},
               'base_06': {'subsample_res': None, 'c_out':  32, 'kernel_res': 0.40, 'concat': False},
               'base_07': {'subsample_res': 0.40, 'c_out':  32, 'kernel_res': 0.40, 'concat': True},
               'base_08': {'subsample_res': None, 'c_out':  64, 'kernel_res': [0.60, 0.60, 0.40], 'concat': False},
               'base_09': {'subsample_res': None, 'c_out':  64, 'kernel_res': [0.60, 0.60, 0.40], 'concat': False},
               'base_10': {'subsample_res': None, 'c_out':  64, 'kernel_res': [0.60, 0.60, 0.40], 'concat': True}}

bev_params = {'bev_00': {'kernel_size': 3, 'c_out': 128},
              'bev_01': {'kernel_size': 3, 'c_out': 128},
              'bev_02': {'kernel_size': 1, 'c_out': len(anchor_params) * output_attr}}

refine_params = {'c_out': 128, 'kernel_size': 3, 'padding': 0.}

# ===========================================Training Controls=============================================

gpu_list = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
            8: 0, 9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6, 15: 7}

batch_size_stage1 = 4
init_lr_stage1 = 1e-4
lr_scale_stage1 = False

batch_size_stage2 = 4
init_lr_stage2 = 2e-4
lr_scale_stage2 = False

decay_epochs = 5
lr_decay = 0.5
lr_warm_up = False
weight_decay = 5e-4
valid_interval = 5
xavier = False
stddev = 1e-3
activation = 'relu'
normalization = None
num_worker = 5
weighted = False
total_epoch = 300

# ===========================================Configuration Directory=============================================

local = False
model_file_name = os.path.basename(__file__).split('.')[0] + '.py'
model_file_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(model_file_dir, model_file_name)
