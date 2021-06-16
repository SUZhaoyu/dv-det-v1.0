import tensorflow as tf
import numpy as np

from models.tf_ops.loader import roi_logits_to_attrs, bbox_logits_to_attrs, dense_voxelization
from models.tf_ops.loader import get_iou_matrix


def roi_logits_to_attrs_tf(base_coors, input_logits, anchor_size):
    anchor_diag = tf.sqrt(tf.pow(anchor_size[0], 2.) + tf.pow(anchor_size[1], 2.))
    w = tf.clip_by_value(tf.exp(input_logits[:, 0]) * anchor_size[0], 0., 1e7)
    l = tf.clip_by_value(tf.exp(input_logits[:, 1]) * anchor_size[1], 0., 1e7)
    h = tf.clip_by_value(tf.exp(input_logits[:, 2]) * anchor_size[2], 0., 1e7)
    x = tf.clip_by_value(input_logits[:, 3] * anchor_diag + base_coors[:, 0], -1e7, 1e7)
    y = tf.clip_by_value(input_logits[:, 4] * anchor_diag + base_coors[:, 1], -1e7, 1e7)
    z = tf.clip_by_value(input_logits[:, 5] * anchor_size[2] + base_coors[:, 2], -1e7, 1e7)
    r = tf.clip_by_value(input_logits[:, 6] * 3.1415927, -1e7, 1e7)
    # r = input_logits[:, 6]
    return tf.stack([w, l, h, x, y, z, r], axis=-1)


def bbox_logits_to_attrs_tf(input_roi_attrs, input_logits):
    roi_diag = tf.sqrt(tf.pow(input_roi_attrs[:, 0], 2.) + tf.pow(input_roi_attrs[:, 1], 2.))
    w = tf.clip_by_value(tf.exp(input_logits[:, 0]) * input_roi_attrs[:, 0], 0., 1e7)
    l = tf.clip_by_value(tf.exp(input_logits[:, 1]) * input_roi_attrs[:, 1], 0., 1e7)
    h = tf.clip_by_value(tf.exp(input_logits[:, 2]) * input_roi_attrs[:, 2], 0., 1e7)
    x = tf.clip_by_value(input_logits[:, 3] * roi_diag + input_roi_attrs[:, 3], -1e7, 1e7)
    y = tf.clip_by_value(input_logits[:, 4] * roi_diag + input_roi_attrs[:, 4], -1e7, 1e7)
    z = tf.clip_by_value(input_logits[:, 5] * input_roi_attrs[:, 2] + input_roi_attrs[:, 5], -1e7, 1e7)
    r = tf.clip_by_value(input_logits[:, 6] * 3.1415927 + input_roi_attrs[:, 6], -1e7, 1e7)
    # r = input_logits[:, 6] + input_roi_attrs[:, 6]
    return tf.stack([w, l, h, x, y, z, r], axis=-1)


def get_roi_attrs(input_logits, base_coors, anchor_size, is_eval=False):
    method = roi_logits_to_attrs if is_eval else roi_logits_to_attrs_tf
    roi_attrs = method(input_logits=input_logits,
                       base_coors=base_coors,
                       anchor_size=anchor_size)
    return roi_attrs


def get_bbox_attrs(input_logits, input_roi_attrs, is_eval=False):
    method = bbox_logits_to_attrs if is_eval else bbox_logits_to_attrs_tf
    bbox_attrs = method(input_logits=input_logits,
                        input_roi_attrs=input_roi_attrs)
    return bbox_attrs


def bev_compression(input_coors,
                    input_features,
                    input_num_list,
                    resolution,
                    dimension_params):
    dense_voxels = dense_voxelization(input_coors=input_coors,
                                      input_features=input_features,
                                      input_num_list=input_num_list,
                                      dimension=dimension_params['dimension'],
                                      offset=dimension_params['offset'],
                                      resolution=resolution)  # [b, w, l, h, c]
    dense_voxels_shape = dense_voxels.get_shape()
    bev_img = tf.reshape(dense_voxels, shape=[tf.shape(dense_voxels)[0],
                                              dense_voxels_shape[1],
                                              dense_voxels_shape[2],
                                              dense_voxels_shape[3]*dense_voxels_shape[4]])

    return bev_img


def get_bev_anchor_coors(bev_img, resolution, offset):
    offset = np.array(offset, dtype=np.float32)
    resolution = np.array(resolution, dtype=np.float32)
    bev_img_shape = bev_img.get_shape()
    img_w = bev_img_shape[1]
    img_l = bev_img_shape[2]

    bev_idx = tf.range(img_w * img_l)  # [w*l]
    anchor_coors = tf.cast(tf.stack([bev_idx // img_l, bev_idx % img_l], axis=-1), dtype=tf.float32)  # [w*l, 2] -> [x, y]
    anchor_coors = anchor_coors * resolution[0:2] + resolution[0:2] / 2. - tf.expand_dims(offset[0:2], axis=0)
    # bev_z_coors = tf.zeros(shape=[img_w * img_l, 1]) + height  # [w * l, 1]
    # bev_3d_coors = tf.expand_dims(tf.concat([bev_2d_coors, bev_z_coors], axis=-1), axis=0)  # [1, w*l, 3]
    # bev_3d_coors = tf.expand_dims(bev_2d_coors, axis=0)  # [1, w*l, 2]
    # anchor_coors = tf.tile(bev_3d_coors, [batch_size, 1, 1])

    return anchor_coors


def get_anchors(bev_img, resolution, offset, anchor_params):
    batch_size = tf.shape(bev_img)[0]
    anchor_coors = get_bev_anchor_coors(bev_img, resolution, offset)

    length = anchor_coors.get_shape()[0]
    output_anchor = []
    num_anchor = len(anchor_params)
    for anchor_param in anchor_params:  # [w, l, h, z, r]
        w = tf.ones([length, 1]) * anchor_param[0]
        l = tf.ones([length, 1]) * anchor_param[1]
        h = tf.ones([length, 1]) * anchor_param[2]
        x = tf.ones([length, 1]) * anchor_coors[:, 0:1]
        y = tf.ones([length, 1]) * anchor_coors[:, 1:2]
        z = tf.ones([length, 1]) * anchor_param[3]
        r = tf.ones([length, 1]) * anchor_param[4]

        anchor = tf.concat([w, l, h, x, y, z, r], axis=-1)  # [w*l, 7]
        output_anchor.append(anchor)

    output_anchor = tf.stack(output_anchor, axis=1, name='anchor_stack')  # [w*l, 2, 7]
    output_anchor = tf.reshape(output_anchor, shape=[length * num_anchor, 7])  # [w*l*2, 7]
    output_anchor = tf.expand_dims(output_anchor, axis=0)  # [1, w*l*2, 7]
    output_anchor = tf.tile(output_anchor, [batch_size, 1, 1])  # [n, w*l*2, 7]
    anchor_num_list = tf.ones([batch_size], dtype=tf.int32) * num_anchor * length

    return output_anchor, anchor_num_list


def get_anchor_ious(anchors, labels):
    '''
    Calculate the IoUs between anchors and labels for the positive anchor selection.
    :param anchors: 3-D Tensor with shape [batch, w*l*2, 7]
    :param labels: 3-D Tensor with shape [batch, 256, 7]
    :return: A 2-D IoU matrix with shape [batch, w*l*2, 256]
    '''
    anchor_ious = get_iou_matrix(input_bbox=anchors,
                                 target_bbox=labels,
                                 ignore_height=True)
    # anchor_ious = tf.reshape(anchor_ious, shape=[-1, anchor_ious.shape[-1]])
    return anchor_ious


def get_proposals_from_anchors(input_anchors, input_logits, clip=False):
    input_anchors = tf.reshape(input_anchors, shape=[-1, input_anchors.shape[-1]])
    anchor_diag = tf.sqrt(tf.pow(input_anchors[:, 0], 2.) + tf.pow(input_anchors[:, 1], 2.))
    if clip:
        w = tf.clip_by_value(tf.exp(input_logits[:, 0]) * input_anchors[:, 0], 0., 1e7)
        l = tf.clip_by_value(tf.exp(input_logits[:, 1]) * input_anchors[:, 1], 0., 1e7)
        h = tf.clip_by_value(tf.exp(input_logits[:, 2]) * input_anchors[:, 2], 0., 1e7)
        x = tf.clip_by_value(input_logits[:, 3] * anchor_diag + input_anchors[:, 3], -1e7, 1e7)
        y = tf.clip_by_value(input_logits[:, 4] * anchor_diag + input_anchors[:, 4], -1e7, 1e7)
        z = tf.clip_by_value(input_logits[:, 5] * input_anchors[:, 2] + input_anchors[:, 5], -1e7, 1e7)
        r = tf.clip_by_value(input_logits[:, 6] * np.pi + input_anchors[:, 6], -1e7, 1e7)
    else:
        w = tf.exp(input_logits[:, 0]) * input_anchors[:, 0]
        l = tf.exp(input_logits[:, 1]) * input_anchors[:, 1]
        h = tf.exp(input_logits[:, 2]) * input_anchors[:, 2]
        x = input_logits[:, 3] * anchor_diag + input_anchors[:, 3]
        y = input_logits[:, 4] * anchor_diag + input_anchors[:, 4]
        z = input_logits[:, 5] * input_anchors[:, 2] + input_anchors[:, 5]
        r = input_logits[:, 6] * np.pi + input_anchors[:, 6]

    return tf.stack([w, l, h, x, y, z, r], axis=-1)


def get_anchor_masks(anchor_ious, low_thres=0.35, high_thres=0.6):
    batch_size = anchor_ious.shape[0]  # b
    num_anchors = anchor_ious.shape[1]  # n
    masks = tf.zeros(shape=[batch_size * num_anchors])  # [b * n]

    matched_anchor_ious = tf.reshape(tf.argmax(anchor_ious, axis=2), shape=[-1])  # [b*n]
    positive_idx = tf.where(tf.greater_equal(matched_anchor_ious, high_thres))
    masks = tf.scatter_nd_update(masks, positive_idx, tf.ones_like(positive_idx))
    ignore_idx = tf.where(tf.logical_and(tf.less(matched_anchor_ious, high_thres), tf.greater(matched_anchor_ious, low_thres)))
    masks = tf.scatter_nd_update(masks, ignore_idx, tf.ones_like(ignore_idx) * -1)

    max_match_idx = tf.argmax(anchor_ious, axis=1)  # [b, k] in (0, n)
    batch_idx_offset = tf.expand_dims(tf.range(batch_size) * num_anchors, axis=-1)  # [b, 1]
    max_match_idx = tf.reshape(max_match_idx + batch_idx_offset, shape=[-1])  # [b*k]
    masks = tf.scatter_nd_update(masks, max_match_idx, tf.ones_like(max_match_idx))  # in {-1, 0, 1}

    return masks






