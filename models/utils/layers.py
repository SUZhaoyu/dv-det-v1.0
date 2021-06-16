from __future__ import division

import tensorflow as tf
import numpy as np

from models.tf_ops.loader import grid_sampling, voxel_sampling_idx_binary, voxel_sampling_idx, voxel_sampling_feature
from models.utils.conv import kernel_conv_wrapper, conv_1d_wrapper, dense_conv_wrapper, conv_3d_wrapper, conv_2d_wrapper

def point_conv_concat(input_coors,
                      input_features,
                      concat_features,
                      input_num_list,
                      voxel_idx,
                      center_idx,
                      layer_params,
                      dimension_params,
                      grid_buffer_size,
                      output_pooling_size,
                      scope,
                      is_training,
                      mem_saving,
                      model_params,
                      trainable=True,
                      bn_decay=None,
                      histogram=False,
                      summary=False,
                      last_layer=False):
    bn_decay = bn_decay if not last_layer else None
    activation = model_params['activation'] if not last_layer else None
    grid_sampling_method = grid_sampling  # TODO: May change with grid_sampling_thrust accordingly.
    voxel_sampling_idx_method = voxel_sampling_idx_binary if mem_saving else voxel_sampling_idx

    if layer_params['subsample_res'] is not None:
        kernel_center_coors, center_num_list, center_idx = \
            grid_sampling_method(input_coors=input_coors,
                                 input_num_list=input_num_list,
                                 resolution=layer_params['subsample_res'],
                                 dimension=dimension_params['dimension'],
                                 offset=dimension_params['offset'])
        if concat_features is not None:
            concat_features = tf.gather(concat_features, center_idx, axis=0)
    else:
        kernel_center_coors = input_coors
        center_num_list = input_num_list
        center_idx = center_idx

    if layer_params['kernel_res'] is not None:
        voxel_idx, _, features = voxel_sampling_idx_method(input_coors=input_coors,
                                                           input_features=input_features,
                                                           input_num_list=input_num_list,
                                                           center_coors=kernel_center_coors,
                                                           center_num_list=center_num_list,
                                                           resolution=layer_params['kernel_res'],
                                                           dimension=dimension_params['dimension'],
                                                           offset=dimension_params['offset'],
                                                           grid_buffer_size=grid_buffer_size,
                                                           output_pooling_size=output_pooling_size,
                                                           with_rpn=False)
    else:
        if layer_params['subsample_res'] is not None:
            voxel_idx = tf.gather(voxel_idx, center_idx, axis=0)
            features = input_features
        else:
            voxel_idx = voxel_idx
            features = input_features


    voxel_features = voxel_sampling_feature(input_features=features,
                                            output_idx=voxel_idx,
                                            padding=model_params['padding'])

    output_features = kernel_conv_wrapper(inputs=voxel_features,
                                          num_output_channels=layer_params['c_out'],
                                          scope=scope,
                                          trainable=trainable,
                                          use_xavier=model_params['xavier'],
                                          stddev=model_params['stddev'],
                                          activation=activation,
                                          bn_decay=bn_decay,
                                          is_training=is_training,
                                          histogram=histogram,
                                          summary=summary)
    if layer_params['concat']:
        if concat_features is None:
            concat_features = output_features
        else:
            concat_features = tf.concat([concat_features, output_features], axis=1)

    return kernel_center_coors, output_features, center_num_list, voxel_idx, center_idx, concat_features


def conv_3d(input_voxels,
            layer_params,
            scope,
            is_training,
            model_params,
            mem_saving,
            trainable=True,
            bn_decay=None,
            histogram=False,
            summary=False,
            last_layer=False):
    bn_decay = bn_decay if not last_layer else None
    activation = model_params['activation'] if not last_layer else None
    conv3d_method = conv_3d_wrapper if mem_saving else dense_conv_wrapper
    output_features = conv3d_method(inputs=input_voxels,
                                    num_output_channels=layer_params['c_out'],
                                    kernel_size=layer_params['kernel_size'],
                                    scope=scope,
                                    trainable=trainable,
                                    use_xavier=model_params['xavier'],
                                    stddev=model_params['stddev'],
                                    activation=activation,
                                    bn_decay=bn_decay,
                                    is_training=is_training,
                                    histogram=histogram,
                                    summary=summary)

    return output_features


def conv_1d(input_points,
            num_output_channels,
            drop_rate,
            model_params,
            scope,
            is_training,
            trainable=True,
            bn_decay=None,
            histogram=False,
            summary=False,
            second_last_layer=False,
            last_layer=False):
    inputs = tf.nn.dropout(input_points, rate=drop_rate)
    activation = model_params['activation'] if not last_layer else None
    if last_layer or second_last_layer:
        bn_decay = None
    output_points = conv_1d_wrapper(inputs=inputs,
                                    num_output_channels=num_output_channels,
                                    scope=scope,
                                    use_xavier=model_params['xavier'],
                                    stddev=model_params['stddev'],
                                    activation=activation,
                                    bn_decay=bn_decay,
                                    is_training=is_training,
                                    trainable=trainable,
                                    histogram=histogram,
                                    summary=summary)
    return output_points


def conv_2d(input_img,
            num_output_channels,
            model_params,
            scope,
            is_training,
            kernel_size=3,
            trainable=True,
            bn_decay=None,
            histogram=False,
            summary=False,
            second_last_layer=False,
            last_layer=False):
    activation = model_params['activation'] if not last_layer else None
    if last_layer or second_last_layer:
        bn_decay = None
    output_points = conv_2d_wrapper(inputs=input_img,
                                    num_output_channels=num_output_channels,
                                    kernel_size=kernel_size,
                                    scope=scope,
                                    use_xavier=model_params['xavier'],
                                    stddev=model_params['stddev'],
                                    activation=activation,
                                    bn_decay=bn_decay,
                                    is_training=is_training,
                                    trainable=trainable,
                                    histogram=histogram,
                                    summary=summary)
    return output_points
