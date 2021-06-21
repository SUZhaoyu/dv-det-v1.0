import os
import sys
sys.path.append('/home/tan/tony/dv-det-v1.0')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from os.path import join
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy

from data.generator.kitti_generator import Dataset
import configs.kitti.kitti_config_training as config
from models.builder.kitti.model_stage1 import inputs_placeholder
from models.tf_ops.loader import grid_sampling, get_gt_bbox
from models.utils.funcs import get_anchors, get_anchor_ious, bev_compression, get_iou_masks, merge_batch_anchors, \
    correct_ignored_masks

from models.tf_ops.test.test_utils import fetch_instance, plot_points


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 4
id = 1
dimension = [100., 140., 9.]
offset = [10., 70., 5.]
anchor_size = [1.6, 3.9, 1.5]
bev_resolution = [0.4, 0.4, 0.8]

DIMENSION_PARAMS = {'dimension': config.dimension,
                    'offset': config.offset}

anchor_param_list = [[1.6, 3.9, 1.5, -1.0, 0.],
                     [1.6, 3.9, 1.5, -1.0, np.pi/2]]

if __name__ == '__main__':
    Dataset = Dataset(task='training',
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
    coors, num_list, idx = grid_sampling(coors, input_num_list_p, 0.5, offset=offset, dimension=dimension)
    features = tf.gather(input_features_p, idx)

    bev_img = bev_compression(input_coors=coors,
                              input_features=features,
                              input_num_list=num_list,
                              resolution=bev_resolution,
                              dimension_params=DIMENSION_PARAMS)

    anchors = get_anchors(bev_img=bev_img,
                          resolution=bev_resolution,
                          offset=DIMENSION_PARAMS['offset'],
                          anchor_params=anchor_param_list)

    anchor_ious = get_anchor_ious(anchors, input_bbox_p[..., :7])
    anchor_iou_masks = get_iou_masks(anchor_ious, 0.35, 0.6)
    anchors, anchor_num_list = merge_batch_anchors(anchors)

    gt_bbox, gt_conf = get_gt_bbox(input_coors=anchors[:, 3:6],
                                   input_num_list=anchor_num_list,
                                   bboxes=input_bbox_p,
                                   padding_offset=0.2,
                                   diff_thres=3,
                                   cls_thres=0,
                                   ignore_height=True)

    anchor_masks = correct_ignored_masks(anchor_iou_masks, gt_conf)
    with tf.Session() as sess:
        for _ in tqdm(range(10)):
            input_coors, input_features, input_num_list, input_labels = next(Dataset.train_generator())
            output_anchors, output_anchor_masks, output_anchor_num_list, output_gt_conf = \
                sess.run([anchors, anchor_masks, anchor_num_list, gt_conf],
                           feed_dict={input_coors_p: input_coors,
                                      input_features_p: input_features,
                                      input_num_list_p: input_num_list,
                                      input_bbox_p: input_labels})

    input_coors = fetch_instance(input_coors, input_num_list, id)
    anchor_masks = fetch_instance(output_anchor_masks, output_anchor_num_list, id)
    anchors = fetch_instance(output_anchors, output_anchor_num_list, id)
    gt_conf = fetch_instance(output_gt_conf, output_anchor_num_list, id)

    output_anchors = anchors[anchor_masks == 1, :]

    plot_points(coors=input_coors,
                bboxes=output_anchors,
                name='anchor_ious_det')






