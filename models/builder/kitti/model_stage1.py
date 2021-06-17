import tensorflow as tf
import configs.kitti.kitti_config_training as CONFIG

from models.tf_ops.loader import get_gt_bbox
from models.utils.layers import point_conv_concat, conv_2d
from models.utils.funcs import bev_compression, get_anchors, get_proposals_from_anchors, get_anchor_ious, \
    get_iou_masks, merge_batch_anchors

ANCHOR_SIZE = CONFIG.anchor_size
EPS = tf.constant(1e-6)

MODEL_PARAMS = {'xavier': CONFIG.xavier,
                'stddev': CONFIG.stddev,
                'activation': CONFIG.activation,
                'padding': -0.5}  # FIXME: need to change the paddings for initial convolution.

DIMENSION_PARAMS = {'dimension': CONFIG.dimension,
                    'offset': CONFIG.offset}


BASE_MODEL_PARAMS = CONFIG.base_params
BEV_RESOLUTION = CONFIG.bev_resolution
BEV_MODEL_PARAMS = CONFIG.bev_params
ANCHOR_PARAMS = CONFIG.anchor_params



def inputs_placeholder(input_channels=1,
                       bbox_padding_num=CONFIG.aug_config['nbbox'],
                       batch_size=None):
    input_coors_p = tf.placeholder(tf.float32, shape=[None, 3], name='stage1_input_coors_p')
    input_features_p = tf.placeholder(tf.float32, shape=[None, input_channels], name='stage1_input_features_p')
    input_num_list_p = tf.placeholder(tf.int32, shape=[batch_size], name='stage1_input_num_list_p')
    input_bbox_p = tf.placeholder(dtype=tf.float32, shape=[batch_size, bbox_padding_num, 9], name='stage1_input_bbox_p')
    return input_coors_p, input_features_p, input_num_list_p, input_bbox_p


def model(input_coors,
          input_features,
          input_num_list,
          is_training,
          trainable,
          mem_saving,
          bn):
    coors, features, num_list = input_coors, input_features, input_num_list
    concat_features, voxel_idx, center_idx = None, None, None

    with tf.variable_scope("stage1"):

        with tf.variable_scope("base"):
            for i, layer_name in enumerate(sorted(BASE_MODEL_PARAMS.keys())):
                coors, features, num_list, voxel_idx, center_idx, concat_features = \
                    point_conv_concat(input_coors=coors,
                                      input_features=features,
                                      concat_features=concat_features,
                                      input_num_list=num_list,
                                      voxel_idx=voxel_idx,
                                      center_idx=center_idx,
                                      layer_params=BASE_MODEL_PARAMS[layer_name],
                                      dimension_params=DIMENSION_PARAMS,
                                      grid_buffer_size=CONFIG.grid_buffer_size,
                                      output_pooling_size=CONFIG.output_pooling_size,
                                      scope="stage1_" + layer_name,
                                      is_training=is_training,
                                      trainable=trainable,
                                      mem_saving=mem_saving,
                                      model_params=MODEL_PARAMS,
                                      bn_decay=bn)

        with tf.variable_scope("bev"):
            bev_img = bev_compression(input_coors=coors,
                                      input_features=features,
                                      input_num_list=num_list,
                                      resolution=BEV_RESOLUTION,
                                      dimension_params=DIMENSION_PARAMS)

            for i, layer_name in enumerate(sorted(BEV_MODEL_PARAMS.keys())):
                bev_img = conv_2d(input_img=bev_img,
                                  kernel_size=BEV_MODEL_PARAMS[layer_name]['kernel_size'],
                                  num_output_channels=BEV_MODEL_PARAMS[layer_name]['c_out'],
                                  model_params=MODEL_PARAMS,
                                  bn_decay=bn,
                                  scope="stage1_" + layer_name,
                                  is_training=is_training,
                                  trainable=trainable,
                                  second_last_layer=(i == len(BEV_MODEL_PARAMS) - 2),
                                  last_layer=(i == len(BEV_MODEL_PARAMS) - 1))

            proposal_logits = tf.reshape(bev_img, shape=[-1, CONFIG.output_attr])
            anchors = get_anchors(bev_img=bev_img,
                                  resolution=BEV_RESOLUTION,
                                  offset=DIMENSION_PARAMS['offset'],
                                  anchor_params=ANCHOR_PARAMS)
            proposals = get_proposals_from_anchors(input_anchors=anchors,
                                                   input_logits=proposal_logits)

    return anchors, proposals


def loss(anchors, proposals, labels):
    anchor_ious = get_anchor_ious(anchors, labels[..., :7])
    anchor_masks = get_iou_masks(anchor_ious=anchor_ious,
                                 low_thres=CONFIG.negative_thres,
                                 high_thres=CONFIG.positive_thres)

    anchors, num_list = merge_batch_anchors(anchors)
    gt_bbox, gt_conf = get_gt_bbox(input_coors=anchors[:, 3:6],
                                   input_num_list=num_list,
                                   bboxes=labels,
                                   padding_offset=CONFIG.padding_offset,
                                   diff_thres=CONFIG.diff_thres,
                                   cls_thres=CONFIG.cls_thres,
                                   ignore_height=True)







