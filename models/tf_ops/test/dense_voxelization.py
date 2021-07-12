import os
import sys
sys.path.append('/home/tan/tony/dv-det-v1.0')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join
import numpy as np
import matplotlib.colors as mcolors
from matplotlib import colors
CSS_COLOR_LIST = list(mcolors.CSS4_COLORS)
import random
import tensorflow as tf
from point_viz.converter import PointvizConverter
Converter = PointvizConverter("/home/tan/tony/threejs")
from tqdm import tqdm

from data.utils.normalization import convert_threejs_coors, convert_threejs_bbox_with_assigned_colors

from data.generator.kitti_generator import KittiDataset
import configs.kitti.kitti_config_training as config
from models.builder.kitti.model_stage1 import inputs_placeholder
from models.tf_ops.loader import grid_sampling, get_gt_bbox, dense_voxelization
from models.utils.funcs import get_anchors, get_anchor_ious, bev_compression, get_iou_masks, merge_batch_anchors, \
    correct_ignored_masks

from models.tf_ops.test.test_utils import fetch_instance, plot_points


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 6
id = 4
anchor_size = [1.6, 3.9, 1.5]
bev_resolution = [0.4, 0.4, 0.8]
low_thres = 0.35
high_thres = 0.6

DIMENSION_PARAMS = {'dimension': config.dimension,
                    'offset': config.offset}

anchor_param_list = [[1.6, 3.9, 1.5, -1.0, 0.],
                     [1.6, 3.9, 1.5, -1.0, np.pi/2]]

def voxels_to_points(input_voxels, resolution, offset):
    w, l, h, c = input_voxels.shape
    output_coors = []
    output_features = []
    for x in tqdm(range(w)):
        for y in range(l):
            for z in range(h):
                if input_voxels[x, y, z, 0] >= 0:
                    output_coors.append([x*resolution[0], y*resolution[1], z*resolution[2]])
                    output_features.append(input_voxels[x, y, z, :])
    return np.array(output_coors) - offset, np.array(output_features)

if __name__ == '__main__':
    Dataset = KittiDataset(task='training',
                           batch_size=batch_size,
                           config=config.aug_config,
                           num_worker=6,
                           hvd_size=1,
                           hvd_id=0)

    input_coors_p, input_features_p, input_num_list_p, input_bbox_p = \
        inputs_placeholder(input_channels=1,
                           bbox_padding_num=config.bbox_padding,
                           batch_size=None)

    coors = tf.cast(input_coors_p, dtype=tf.float32)
    coors, num_list, idx = grid_sampling(coors, input_num_list_p,
                                         resolution=0.1,
                                         dimension=DIMENSION_PARAMS['dimension'],
                                         offset=DIMENSION_PARAMS['offset'])
    features = tf.gather(input_features_p, idx)

    voxels = dense_voxelization(input_coors=coors,
                                input_features=features,
                                input_num_list=num_list,
                                resolution=[0.2, 0.2, 0.4],
                                dimension=DIMENSION_PARAMS['dimension'],
                                offset=DIMENSION_PARAMS['offset'])

    with tf.Session() as sess:
        for _ in tqdm(range(3)):
            input_coors, input_features, input_num_list, input_labels = next(Dataset.train_generator())
            output_voxels = \
                sess.run(voxels,
                           feed_dict={input_coors_p: input_coors,
                                      input_features_p: input_features,
                                      input_num_list_p: input_num_list,
                                      input_bbox_p: input_labels})

        output_voxels = output_voxels[id]
        output_coors, output_features = voxels_to_points(input_voxels=output_voxels,
                                                         resolution=[0.2, 0.2, 0.4],
                                                         offset=DIMENSION_PARAMS['offset'])
        Converter.compile(coors=convert_threejs_coors(output_coors),
                          intensity=output_features[..., 0],
                          default_rgb=None,
                          bbox_params=None,
                          task_name='dense_voxelization')

        Converter.compile(coors=convert_threejs_coors(fetch_instance(input_coors, input_num_list, id)),
                          intensity=fetch_instance(input_features, input_num_list, id)[:, 0],
                          default_rgb=None,
                          bbox_params=None,
                          task_name='dense_voxelization_input')

