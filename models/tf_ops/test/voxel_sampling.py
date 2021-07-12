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


from data.generator.kitti_generator import KittiDataset
import configs.kitti.kitti_config_training as config
from models.builder.kitti.model_stage1 import inputs_placeholder
from models.tf_ops.loader import grid_sampling

from models.tf_ops.test.test_utils import fetch_instance, plot_points


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 1
id = 0
anchor_size = [1.6, 3.9, 1.5]
bev_resolution = [0.4, 0.4, 0.8]
low_thres = 0.35
high_thres = 0.6

DIMENSION_PARAMS = {'dimension': config.dimension,
                    'offset': config.offset}

anchor_param_list = [[1.6, 3.9, 1.5, -1.0, 0.],
                     [1.6, 3.9, 1.5, -1.0, np.pi/2]]

MODEL_PARAMS = {'xavier': config.xavier,
                'stddev': config.stddev,
                'activation': config.activation,
                'padding': -0.5}

if __name__ == '__main__':
    Dataset = KittiDataset(task='validation',
                           batch_size=batch_size,
                           config=None,
                           num_worker=6,
                           hvd_size=1,
                           hvd_id=0)

    input_coors_p, input_features_p, input_num_list_p, input_bbox_p = \
        inputs_placeholder(input_channels=1,
                           bbox_padding_num=config.bbox_padding,
                           batch_size=None)
    is_training_p = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')

    coors = tf.cast(input_coors_p, dtype=tf.float32)

    coors, num_list, idx = grid_sampling(coors, input_num_list_p,
                                         resolution=0.1,
                                         dimension=DIMENSION_PARAMS['dimension'],
                                         offset=DIMENSION_PARAMS['offset'])
    features = tf.gather(input_features_p, idx)


    init_op = tf.initialize_all_variables()

    TF_CONFIG = tf.ConfigProto()
    TF_CONFIG.gpu_options.visible_device_list = "0"
    TF_CONFIG.gpu_options.allow_growth = True
    TF_CONFIG.allow_soft_placement = False
    TF_CONFIG.log_device_placement = False

    with tf.Session(config=TF_CONFIG) as sess:
        sess.run(init_op)
        for _ in tqdm(range(3)):
            input_coors, input_features, input_num_list, input_labels = next(Dataset.train_generator())
            output_coors, output_features, output_num_list = \
                sess.run([coors, num_list, idx],
                           feed_dict={input_coors_p: input_coors,
                                      input_features_p: input_features,
                                      input_num_list_p: input_num_list,
                                      input_bbox_p: input_labels,
                                      is_training_p: False})
            print(output_num_list)

