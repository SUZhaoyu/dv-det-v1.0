import tensorflow as tf
import configs.kitti.kitti_config_training as CONFIG

from models.tf_ops.loader import get_gt_bbox
from models.utils.layers import point_conv_concat, conv_2d
from models.utils.iou import cal_3d_iou
from models.utils.loss import get_masked_average, focal_loss, smooth_l1_loss
from models.utils.funcs import bev_compression, get_anchors, get_proposals_from_anchors, get_anchor_ious, \
    get_iou_masks, merge_batch_anchors, correct_ignored_masks

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
            proposals, pred_conf = get_proposals_from_anchors(input_anchors=anchors,
                                                              input_logits=proposal_logits,
                                                              clip=True)

    return anchors, proposals, pred_conf


def loss(anchors, proposals, pred_conf, labels, weight_decay):
    anchor_ious = get_anchor_ious(anchors, labels[..., :7])
    anchor_iou_masks = get_iou_masks(anchor_ious=anchor_ious,
                                     low_thres=CONFIG.negative_thres,
                                     high_thres=CONFIG.positive_thres,
                                     force_ignore_thres=CONFIG.forge_ignore_thres)

    anchors, num_list = merge_batch_anchors(anchors)
    gt_bbox, gt_conf = get_gt_bbox(input_coors=anchors[:, 3:6],
                                   input_num_list=num_list,
                                   bboxes=labels,
                                   padding_offset=CONFIG.padding_offset,
                                   diff_thres=CONFIG.diff_thres,
                                   cls_thres=CONFIG.cls_thres,
                                   ignore_height=True)

    anchor_masks = correct_ignored_masks(iou_masks=anchor_iou_masks,
                                         gt_conf=gt_conf)
    positive_masks = tf.cast(tf.equal(anchor_masks, 1), dtype=tf.float32)
    conf_masks = tf.cast(tf.greater_equal(anchor_masks, 0), dtype=tf.float32)
    tf.summary.scalar('positive_anchor_count', tf.reduce_sum(positive_masks))

    ious = cal_3d_iou(gt_attrs=gt_bbox,
                      pred_attrs=proposals,
                      clip=False)
    iou_loss = get_masked_average(1. - ious, positive_masks)
    averaged_iou = get_masked_average(ious, positive_masks)
    tf.summary.scalar('stage1_iou_loss', iou_loss)

    angle_l1_loss = smooth_l1_loss(labels=gt_bbox[:, 6],
                                   predictions=proposals[:, 6],
                                   with_sin=True)
    angle_l1_loss = get_masked_average(angle_l1_loss, positive_masks)
    tf.summary.scalar('stage1_angle_l1_loss', angle_l1_loss)
    tf.summary.scalar('stage1_angle_sin_bias', get_masked_average(tf.abs(tf.sin(gt_bbox[:, 6] - proposals[:, 6])), positive_masks))
    tf.summary.scalar('stage1_angle_bias', get_masked_average(tf.abs(gt_bbox[:, 6] - proposals[:, 6]), positive_masks))

    conf_target = tf.cast(gt_conf, dtype=tf.float32) * conf_masks  # [-1, 0, 1] * [0, 1, 1] -> [0, 0, 1]
    conf_loss = get_masked_average(focal_loss(label=conf_target, pred=pred_conf, alpha=0.25), conf_masks)
    tf.summary.scalar('stage1_conf_loss', conf_loss)

    regular_l2_loss = weight_decay * tf.add_n(tf.get_collection("stage1_l2"))
    tf.summary.scalar('stage1_regularization_l2_loss', regular_l2_loss)

    total_loss = 2 * iou_loss + angle_l1_loss + conf_loss + regular_l2_loss

    return total_loss, averaged_iou
