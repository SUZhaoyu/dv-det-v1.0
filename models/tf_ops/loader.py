import os
from os.path import join
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

CWD = os.path.dirname(os.path.abspath(__file__))

# =============================================Grid Sampling===============================================

grid_sampling_exe = tf.load_op_library(join(CWD, 'build', 'grid_sampling.so'))

def grid_sampling(input_coors,
                  input_num_list,
                  resolution,
                  dimension,
                  offset):
    # if type(resolution) is float:
    #     resolution = [resolution] * 3
    output_idx, output_num_list = grid_sampling_exe.grid_sampling_op(input_coors=input_coors + offset,
                                                                     input_num_list=input_num_list,
                                                                     dimension=dimension,
                                                                     resolution=resolution)
    output_coors = tf.gather(input_coors, output_idx, axis=0)
    return output_coors, output_num_list, output_idx

ops.NoGradient("GridSamplingOp")

# =============================================Voxel Sampling Idx===============================================

voxel_sampling_idx_exe = tf.load_op_library(join(CWD, 'build', 'voxel_sampling_idx.so'))


def voxel_sampling_idx(input_coors,
                       input_features,
                       input_num_list,
                       center_coors,
                       center_num_list,
                       resolution,
                       dimension,
                       offset,
                       grid_buffer_size,
                       output_pooling_size,
                       with_rpn=False):
    if type(resolution) is float:
        resolution = [resolution] * 3
    output_idx, valid_idx = voxel_sampling_idx_exe.voxel_sampling_idx_op(input_coors=input_coors + offset,
                                                                         input_num_list=input_num_list,
                                                                         center_coors=center_coors + offset,
                                                                         center_num_list=center_num_list,
                                                                         dimension=dimension,
                                                                         resolution=resolution,
                                                                         grid_buffer_size=grid_buffer_size,
                                                                         output_pooling_size=output_pooling_size,
                                                                         with_rpn=with_rpn)
    return output_idx, valid_idx, input_features


ops.NoGradient("VoxelSamplingIdxOp")


# =============================================Voxel Sampling Idx Binary===============================================

voxel_sampling_idx_binary_exe = tf.load_op_library(join(CWD, 'build', 'voxel_sampling_idx_binary.so'))
def voxel_sampling_idx_binary(input_coors,
                              input_features,
                              input_num_list,
                              center_coors,
                              center_num_list,
                              resolution,
                              dimension,
                              offset,
                              grid_buffer_size,
                              output_pooling_size,
                              with_rpn=False):

    if type(resolution) is float:
        resolution = [resolution] * 3
    npoint = tf.shape(input_coors)[0]
    batch_size = tf.shape(input_num_list)[0]
    dim_w = tf.cast(tf.floor(dimension[0] / resolution[0]), dtype=tf.int64)
    dim_l = tf.cast(tf.floor(dimension[1] / resolution[1]), dtype=tf.int64)
    dim_h = tf.cast(tf.floor(dimension[2] / resolution[2]), dtype=tf.int64)
    dim_offset = dim_w * dim_l * dim_h

    point_ids = tf.range(npoint) + 1
    point_ids_array = tf.cast(tf.tile(tf.expand_dims(point_ids, axis=0), [batch_size, 1]), dtype=tf.float32)
    accu_num_list = tf.cast(tf.cumsum(input_num_list), dtype=tf.float32)
    masks = tf.cast(tf.greater(point_ids_array / tf.expand_dims(accu_num_list, axis=-1), 1.0), dtype=tf.int64)
    voxel_offset_masks = tf.reduce_sum(masks, axis=0) * dim_offset

    input_voxel_coors = tf.cast(tf.floor((input_coors + offset) / resolution), dtype=tf.int64)
    # input_voxel_coors = tf.clip_by_value(input_voxel_coors, clip_value_min=0, clip_value_max=[dim_w - 1, dim_l - 1, dim_h - 1])
    input_voxel_ids = input_voxel_coors[:, 2] * dim_l * dim_w + input_voxel_coors[:, 1] * dim_w + input_voxel_coors[:, 0]
    input_voxel_ids += voxel_offset_masks
    sorted_args = tf.argsort(input_voxel_ids)
    sorted_voxel_ids = tf.gather(input_voxel_ids, sorted_args) - voxel_offset_masks
    sorted_coors = tf.gather(input_coors, sorted_args, axis=0)
    sorted_features = tf.gather(input_features, sorted_args, axis=0)
    # XXX: Need to pay attention to the back-propagation implementation.
    output_idx, valid_idx = voxel_sampling_idx_binary_exe.voxel_sampling_idx_binary_op(input_coors=sorted_coors + offset,
                                                                                       input_voxel_idx=sorted_voxel_ids,
                                                                                       input_num_list=input_num_list,
                                                                                       center_coors=center_coors + offset,
                                                                                       center_num_list=center_num_list,
                                                                                       dimension=dimension,
                                                                                       resolution=resolution,
                                                                                       grid_buffer_size=grid_buffer_size,
                                                                                       output_pooling_size=output_pooling_size,
                                                                                       with_rpn=with_rpn)
    return output_idx, valid_idx, sorted_features

ops.NoGradient("VoxelSamplingIdxBinaryOp")

# =============================================Voxel Sampling Feature===============================================

voxel_sampling_feature_exe = tf.load_op_library(join(CWD, 'build', 'voxel_sampling_feature.so'))


def voxel_sampling_feature(input_features,
                           output_idx,
                           padding):
    output_features = voxel_sampling_feature_exe.voxel_sampling_feature_op(input_features=input_features,
                                                                           output_idx=output_idx,
                                                                           padding_value=padding)
    return output_features

@ops.RegisterGradient("VoxelSamplingFeatureOp")
def voxel_sampling_feature_grad(op, grad):
    input_features = op.inputs[0]
    output_idx = op.inputs[1]
    input_features_grad = voxel_sampling_feature_exe.voxel_sampling_feature_grad_op(input_features=input_features,
                                                                                    output_idx=output_idx,
                                                                                    output_features_grad=grad)
    return [input_features_grad, None]

def voxel_sampling_feature_grad_test(input_features, output_idx, grad):
    input_features_grad = voxel_sampling_feature_exe.voxel_sampling_feature_grad_op(input_features=input_features,
                                                                                    output_idx=output_idx,
                                                                                    output_features_grad=grad)
    return input_features_grad

# =============================================La Roi Pooling Fast===============================================

la_roi_pooling_fast_exe = tf.load_op_library(join(CWD, 'build', 'la_roi_pooling_fast.so'))
def la_roi_pooling_fast(input_coors, input_features, roi_attrs, input_num_list, roi_num_list,
                        dimension, offset, grid_buffer_resolution=0.8,
                        grid_buffer_size=4, voxel_size=5, padding_value=0., pooling_size=5):
    output_features, _, _ = la_roi_pooling_fast_exe.la_roi_pooling_fast_op(input_coors=input_coors + offset,
                                                                           input_features=input_features,
                                                                           roi_attrs=roi_attrs,
                                                                           input_num_list=input_num_list,
                                                                           roi_num_list=roi_num_list,
                                                                           voxel_size=voxel_size,
                                                                           padding_value=padding_value,
                                                                           pooling_size=pooling_size,
                                                                           dimension=dimension,
                                                                           offset=offset,
                                                                           grid_buffer_resolution=grid_buffer_resolution,
                                                                           grid_buffer_size=grid_buffer_size)
    return output_features
# ops.NoGradient("LaRoiPoolingFastOp")

@ops.RegisterGradient("LaRoiPoolingFastOp")
def la_roi_pooling_fast_grad(op, grad, _, __):
    input_features = op.inputs[1]
    output_idx = op.outputs[1]
    output_weight = op.outputs[2]
    input_features_grad = la_roi_pooling_fast_exe.la_roi_pooling_fast_grad_op(input_features=input_features,
                                                                              output_idx=output_idx,
                                                                              output_weight=output_weight,
                                                                              output_features_grad=grad)
    return [None, input_features_grad, None, None, None]

# =============================================Voxel2Col===============================================

voxel_to_col_exe = tf.load_op_library(join(CWD, 'build', 'voxel2col.so'))
def voxel2col(input_voxels, kernel_size=3):
    channels = input_voxels.shape[2]
    output_voxels, _ = voxel_to_col_exe.voxel_to_col_op(input_voxels=input_voxels,
                                                        kernel_size=kernel_size,
                                                        channels=channels)
    return output_voxels

@ops.RegisterGradient("VoxelToColOp")
def voxel2col_grad(op, grad, _):
    input_voxels = op.inputs[0]
    output_idx = op.outputs[1]
    input_voxels_grad = voxel_to_col_exe.voxel_to_col_grad_op(input_voxels=input_voxels,
                                                              output_idx=output_idx,
                                                              output_voxels_grad=grad)
    return input_voxels_grad


# =============================================Dense Voxelization===============================================

dense_voxelization_exe = tf.load_op_library(join(CWD, 'build', 'dense_voxelization.so'))
def dense_voxelization(input_coors, input_features, input_num_list, resolution, dimension, offset):
    if type(resolution) is float:
        resolution = [resolution] * 3
    output_features, idx = dense_voxelization_exe.dense_voxelization_op(input_coors=input_coors + offset,
                                                                      input_features=input_features,
                                                                      input_num_list=input_num_list,
                                                                      resolution=resolution,
                                                                      dimension=dimension)
    return output_features, idx

@ops.RegisterGradient("DenseVoxelizationOp")
def dense_voxelization_grad(op, grad, _):
    input_features = op.inputs[1]
    output_idx = op.outputs[1]
    input_features_grad = dense_voxelization_exe.dense_voxelization_grad_op(input_features=input_features,
                                                                            output_idx=output_idx,
                                                                            output_features_grad=grad)
    return [None, input_features_grad, None]

def dense_voxelization_grad_test(input_features, output_idx, grad):

    input_features_grad = dense_voxelization_exe.dense_voxelization_grad_op(input_features=input_features,
                                                                            output_idx=output_idx,
                                                                            output_features_grad=grad)
    return input_features_grad

# =============================================RoI Filter===============================================

roi_filter_exe = tf.load_op_library(join(CWD, 'build', 'roi_filter.so'))
def roi_filter(input_roi_attrs, input_roi_conf, input_roi_ious, input_num_list, conf_thres, iou_thres, max_length, with_negative):
    output_num_list, output_idx = roi_filter_exe.roi_filter_op(input_roi_conf=input_roi_conf,
                                                               input_roi_ious=input_roi_ious,
                                                               input_num_list=input_num_list,
                                                               conf_thres=conf_thres,
                                                               iou_thres=iou_thres,
                                                               max_length=max_length,
                                                               with_negative=with_negative)
    output_roi_attrs = tf.gather(input_roi_attrs, output_idx, axis=0)
    return output_roi_attrs, output_num_list, output_idx
ops.NoGradient("RoiFilterOp")

# =============================================Get Ground Truth Bbox===============================================

get_gt_bbox_exe = tf.load_op_library(join(CWD, 'build', 'get_gt_bbox.so'))


def get_gt_bbox(input_coors, bboxes, input_num_list, padding_offset=0.2, diff_thres=3, cls_thres=0, ignore_height=False):
    '''
    Get point-wise ground truth. according to the point coordinates.
    :param input_coors: 2-D Tensor with shape [npoint, 3]
    :param bboxes: 3-D Tensor with shape [batch, nbbox, bbox_attr]
    :param input_num_list: 1-D Tensor with shape [batch]
    :param padding_offset: default=0.2
    :param diff_thres: default=3, only the points with difficulty <= diff_thres will be linked to the final loss
    :param cls_thres: default=0, only the points with class <= cls_thres will be linked to the final loss
    :param ignore_height: default=False, whether to consider the height location. Set it to False for BEV regression,
                          and True for the second stage proposal refinement.
    :return: gt_attrs: 2-D Tensor with shape [npoint, 7]: [w, l, h, offset_x, offset_y, offset_z, angle]
             gt_conf: 1-D Tensor with shape [npoint], indicating the status of the returned gt_attrs
             {0: Background; 1: Foreground; -1: Ignore}

    '''
    gt_attrs, gt_conf = get_gt_bbox_exe.get_gt_bbox_op(input_coors=input_coors,
                                                       gt_bbox=bboxes,
                                                       input_num_list=input_num_list,
                                                       padding_offset=padding_offset,
                                                       diff_thres=diff_thres,
                                                       cls_thres=cls_thres,
                                                       ignore_height=ignore_height)
    return gt_attrs, gt_conf
ops.NoGradient("GetGtBboxOp")

# =============================================Arg Sort===============================================

arg_sort_exe = tf.load_op_library(join(CWD, 'build', 'arg_sort.so'))
def arg_sort(inputs, descending=True):
    ret_idx = arg_sort_exe.arg_sort_op(input=inputs,
                                       descending=descending)
    return ret_idx
ops.NoGradient("ArgSortOp")

# =============================================Bbox IoU===============================================

nms_exe = tf.load_op_library(join(CWD, 'build', 'nms.so'))
def get_iou_matrix(input_bbox, target_bbox, ignore_height=False):
    iou_matrix = nms_exe.boxes_iou_op(input_boxes_a=input_bbox,
                                      input_boxes_b=target_bbox,
                                      ignore_height=ignore_height)
    return iou_matrix
ops.NoGradient("BoxesIouOp")