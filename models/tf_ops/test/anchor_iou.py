import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from os.path import join
import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.python.client import timeline
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy

from data.generator.kitti_generator import Dataset
import configs.kitti.kitti_config_training as config
tf.enable_eager_execution()
from models.tf_ops.loader import grid_sampling, dense_voxelization
from models.utils.funcs import get_anchors, get_anchor_ious, bev_compression

from models.tf_ops.test.test_utils import fetch_instance, plot_points


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 6
id = 3
dimension = [100., 140., 9.]
offset = [10., 70., 5.]
anchor_size = [1.6, 3.9, 1.5]

DIMENSION_PARAMS = {'dimension': config.dimension_training,
                    'offset': config.offset_training}

anchor_param_list = [[1.6, 3.9, 1.5, -1.0, 0.],
                     [1.6, 3.9, 1.5, -1.0, np.pi/2]]

if __name__ == '__main__':
    Dataset = Dataset(task='training',
                      batch_size=batch_size,
                      config=config.aug_config,
                      num_worker=6,
                      hvd_size=1,
                      hvd_id=0)
    input_coors, input_features, input_num_list, input_labels = next(Dataset.train_generator())
    Dataset.stop()

    coors = tf.cast(input_coors, dtype=tf.float32)
    coors, num_list, idx = grid_sampling(coors, input_num_list, 0.5, offset=offset, dimension=dimension)
    features = tf.gather(input_features, idx)

    bev_img = bev_compression(input_coors=coors,
                              input_features=features,
                              input_num_list=num_list,
                              resolution=[0.6, 0.6, 0.8],
                              dimension_params=DIMENSION_PARAMS)

    anchors = get_anchors(bev_img=bev_img,
                          resolution=[0.6, 0.6, 0.8],
                          offset=DIMENSION_PARAMS['offset'],
                          anchor_params=anchor_param_list)

    anchor_ious = get_anchor_ious(anchors, input_labels[..., :7])

    anchors = anchors.numpy()[id]
    anchor_ious = anchor_ious.numpy()[id] # [n, 256]
    input_labels = input_labels[id]
    input_points = fetch_instance(input_coors, input_num_list, id)


    # anchor_ious = np.max(anchor_ious, axis=1) # [n]

    max_match_anchor_ious = np.argmax(anchor_ious, axis=0)  # [256]
    max_match_anchors = anchors[max_match_anchor_ious]

    positive_anchors = anchors[np.max(anchor_ious, axis=1) > 0.6]

    output_anchors = np.concatenate([max_match_anchors, positive_anchors], axis=0)

    # positive_anchors = anchors[np.max(anchor_ious, axis=1) > 0.6]

    plot_points(coors=input_points,
                bboxes=output_anchors,
                name='anchor_ious')






