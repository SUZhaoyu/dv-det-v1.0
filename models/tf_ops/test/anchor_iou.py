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
from models.tf_ops.loader import grid_sampling, get_gt_bbox
from models.utils.funcs import get_anchors, get_anchor_ious, bev_compression, get_iou_masks, merge_batch_anchors, \
    correct_ignored_masks

from models.tf_ops.test.test_utils import fetch_instance, plot_points


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 1
id = 0
anchor_size = [1.6, 3.9, 1.5]
bev_resolution = [0.4, 0.4, 0.8]
low_thres = 0.05
high_thres = 0.2

DIMENSION_PARAMS = {'dimension': config.dimension,
                    'offset': config.offset}

anchor_param_list = [[1.6, 3.9, 1.5, -1.0, 0.],
                     [1.6, 3.9, 1.5, -1.0, np.pi/2]]

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
                                         resolution=0.5,
                                         dimension=DIMENSION_PARAMS['dimension'],
                                         offset=DIMENSION_PARAMS['offset'])
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
    anchor_iou_masks = get_iou_masks(anchor_ious, low_thres=low_thres, high_thres=high_thres)

    anchor_ious = tf.reshape(tf.reduce_max(anchor_ious, axis=2), shape=[-1])  # [b*n]

    anchors, anchor_num_list = merge_batch_anchors(anchors)

    gt_bbox, gt_conf = get_gt_bbox(input_coors=anchors[:, 3:6],
                                   input_num_list=anchor_num_list,
                                   bboxes=input_bbox_p,
                                   padding_offset=0.2,
                                   diff_thres=3,
                                   cls_thres=0,
                                   ignore_height=True)

    # anchor_masks = correct_ignored_masks(anchor_iou_masks, gt_conf)
    with tf.Session() as sess:
        for _ in tqdm(range(3)):
            input_coors, input_features, input_num_list, input_labels = next(Dataset.train_generator())
            output_anchors, output_anchor_ious, output_anchor_masks, output_anchor_num_list, output_gt_bbox, output_gt_conf = \
                sess.run([anchors, anchor_ious, anchor_iou_masks, anchor_num_list, gt_bbox, gt_conf],
                           feed_dict={input_coors_p: input_coors,
                                      input_features_p: input_features,
                                      input_num_list_p: input_num_list,
                                      input_bbox_p: input_labels})

    input_coors = fetch_instance(input_coors, input_num_list, id)
    output_anchor_masks = fetch_instance(output_anchor_masks, output_anchor_num_list, id)
    output_anchor_ious = fetch_instance(output_anchor_ious, output_anchor_num_list, id)
    output_anchors = fetch_instance(output_anchors, output_anchor_num_list, id)
    output_gt_conf = fetch_instance(output_gt_conf, output_anchor_num_list, id)
    output_gt_bbox = fetch_instance(output_gt_bbox, output_anchor_num_list, id)


    positive_masks = np.squeeze(np.argwhere(output_anchor_masks == 1))
    negative_masks = np.squeeze(np.argwhere(output_anchor_masks == 0))
    ignore_masks = np.squeeze(np.argwhere(output_anchor_masks == -1))

    input_rgbs = np.zeros_like(input_coors) + [255, 0, 255]
    anchor_coors = output_anchors[:, 3:6]
    anchor_rgbs = np.zeros_like(anchor_coors)
    anchor_rgbs[positive_masks, :] += [255, 255, 255]
    anchor_rgbs[negative_masks, :] += [64, 64, 64]
    anchor_rgbs[ignore_masks, :] += [128, 128, 128]

    output_coors = np.concatenate([input_coors, anchor_coors], axis=0)
    output_rgbs = np.concatenate([input_rgbs, anchor_rgbs], axis=0)


    # output_gt_bbox = output_gt_bbox[output_anchor_masks == 1, :]
    # positive_anchors = output_anchors[output_anchor_masks == 1, :]
    # positive_anchor_coors = output_anchors[output_anchor_masks == 1, 3:6]
    # negative_anchor_coors = output_anchors[output_anchor_masks == 0, 3:6]
    # ignore_anchor_coors = output_anchors[output_anchor_masks == -1, 3:6]
    #
    # output_color_names = []
    # output_color_rgbs = []
    # for i in range(len(positive_anchor_coors)):
    #     color_name = random.choice(CSS_COLOR_LIST)
    #     output_color_names.append(color_name)
    #     output_color_rgbs.append(np.array(colors.to_rgb(color_name)) * 255)
    #
    #
    # output_coors = np.concatenate([input_coors, negative_anchor_coors, ignore_anchor_coors], axis=0)
    # input_rgbs = np.zeros_like(input_coors) + [255, 255, 255]
    # positive_rgbs = np.array(output_color_rgbs)
    # negative_rgbs = np.zeros_like(negative_anchor_coors)
    # ignore_rgbs = np.zeros_like(ignore_anchor_coors) + [128, 128, 128]
    # output_rgbs = np.concatenate([input_rgbs, negative_rgbs, ignore_rgbs], axis=0)
    #
    # # output_coors = output_anchors[:, 3:6]
    # # output_rgbs = np.expand_dims((output_anchor_masks + 1), axis=-1) * (np.zeros_like(output_coors) + [127, 127, 127])
    # # output_rgbs = np.expand_dims((output_anchor_masks), axis=-1) * (np.zeros_like(output_coors) + [255, 255, 255])
    #
    #
    #
    # gt_bbox_params = convert_threejs_bbox_with_assigned_colors(output_gt_bbox, colors=output_color_names) if len(output_gt_bbox) > 0 else []
    # anchor_params = convert_threejs_bbox_with_assigned_colors(positive_anchors, colors=output_color_names) if len(positive_anchors) > 0 else []
    # bbox_params = anchor_params

    Converter.compile(coors=convert_threejs_coors(output_coors),
                      intensity=None,
                      default_rgb=output_rgbs,
                      bbox_params=None,
                      task_name='anchor_iou_det-1')
