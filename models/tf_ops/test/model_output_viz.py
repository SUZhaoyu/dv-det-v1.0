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
import configs.kitti.kitti_config_training as CONFIG
from models.tf_ops.loader import grid_sampling, get_gt_bbox
from models.utils.funcs import get_anchors, get_anchor_ious, bev_compression, get_iou_masks, merge_batch_anchors, \
    correct_ignored_masks

from models.tf_ops.test.test_utils import fetch_instance, plot_points
from models.builder.kitti import model_stage1 as MODEL


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 4
id = 3
dimension = [100., 140., 9.]
offset = [10., 70., 5.]
anchor_size = [1.6, 3.9, 1.5]
bev_resolution = [0.4, 0.4, 0.8]

DIMENSION_PARAMS = {'dimension': CONFIG.dimension,
                    'offset': CONFIG.offset}

anchor_param_list = [[1.6, 3.9, 1.5, -1.0, 0.],
                     [1.6, 3.9, 1.5, -1.0, np.pi/2]]

if __name__ == '__main__':
    Dataset = Dataset(task='training',
                      batch_size=batch_size,
                      config=CONFIG.aug_config,
                      num_worker=6,
                      hvd_size=1,
                      hvd_id=0)

    input_coors_p, input_features_p, input_num_list_p, input_bbox_p = \
        MODEL.inputs_placeholder(input_channels=1,
                                 bbox_padding_num=CONFIG.bbox_padding,
                                 batch_size=None)

    anchors, proposals, pred_conf = MODEL.model(input_coors=input_coors_p,
                                                input_features=input_features_p,
                                                input_num_list=input_num_list_p,
                                                is_training=True,
                                                trainable=True,
                                                mem_saving=True,
                                                bn=1.)

    anchor_ious = get_anchor_ious(anchors, input_bbox_p[..., :7])
    anchor_iou_masks = get_iou_masks(anchor_ious=anchor_ious,
                                     low_thres=CONFIG.negative_thres,
                                     high_thres=CONFIG.positive_thres,
                                     force_ignore_thres=CONFIG.forge_ignore_thres)

    anchors, num_list = merge_batch_anchors(anchors)
    gt_bbox, gt_conf = get_gt_bbox(input_coors=anchors[:, 3:6],
                                   input_num_list=num_list,
                                   bboxes=input_bbox_p,
                                   padding_offset=CONFIG.padding_offset,
                                   diff_thres=3,
                                   cls_thres=CONFIG.cls_thres,
                                   ignore_height=True)

    anchor_masks = correct_ignored_masks(iou_masks=anchor_iou_masks,
                                         gt_conf=gt_conf)

    init_op = tf.initialize_all_variables()


    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer(), feed_dict={input_num_list_p: input_num_list})
        sess.run(init_op)
        for _ in tqdm(range(10)):
            input_coors, input_features, input_num_list, input_labels = next(Dataset.train_generator())
            output_anchors, output_anchor_masks, output_anchor_num_list, output_gt_bbox, output_gt_conf = \
                sess.run([anchors, anchor_masks, num_list, gt_bbox, gt_conf],
                           feed_dict={input_coors_p: input_coors,
                                      input_features_p: input_features,
                                      input_num_list_p: input_num_list,
                                      input_bbox_p: input_labels})

    # anchor_num_list = anchor_num_list.numpy()
    # anchor_masks = anchor_masks.numpy()
    # gt_conf = gt_conf.numpy()
    # anchors = anchors.numpy()

    input_coors = fetch_instance(input_coors, input_num_list, id)
    anchor_masks = fetch_instance(output_anchor_masks, output_anchor_num_list, id)
    anchors = fetch_instance(output_anchors, output_anchor_num_list, id)
    gt_conf = fetch_instance(output_gt_conf, output_anchor_num_list, id)
    output_gt_bbox = fetch_instance(output_gt_bbox, output_anchor_num_list, id)

    output_anchors = output_gt_bbox[anchor_masks == 1, :]

    plot_points(coors=input_coors,
                bboxes=output_anchors,
                name='anchor_ious_det')
