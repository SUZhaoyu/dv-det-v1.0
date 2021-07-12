import os
from os.path import join
import sys
sys.path.append("/home/tan/tony/dv-det-v1.0")
import numpy as np
import tensorflow as tf
from data.utils.normalization import convert_threejs_coors, convert_threejs_bbox_with_colors, convert_threejs_bbox_with_prob, get_points_rgb_with_prob
import horovod.tensorflow as hvd
from point_viz.converter import PointvizConverter
from tensorflow.python.client import timeline

from tqdm import tqdm
import configs.kitti.kitti_config_inference as CONFIG

os.system("rm -r {}".format('/home/tan/tony/threejs/kitti-stage1'))
Converter = PointvizConverter(home='/home/tan/tony/threejs/kitti-stage1')

from models.builder.kitti import model_stage1 as MODEL

anchor_size = CONFIG.anchor_size
# model_path = '/home/tan/tony/dv-det/ckpt-kitti/stage1-test/test/model_0.7087964968982694'
# model_path = '/home/tan/tony/dv-det-v1.0/ckpt-kitti/test/test/model_0.5969989594351234'
data_home = '/home/tan/tony/dv-det-v1.0/eval/kitti/data'
visualization = True
task = 'validation'

input_coors_stack = np.load(join(data_home, 'input_coors_{}.npy'.format(task)), allow_pickle=True)
input_features_stack = np.load(join(data_home, 'input_features_{}.npy'.format(task)), allow_pickle=True)
input_num_list_stack = np.load(join(data_home, 'input_num_list_{}.npy'.format(task)), allow_pickle=True)
input_bboxes_stack = np.load(join(data_home, 'input_bboxes_{}.npy'.format(task)), allow_pickle=True)

input_coors_p, input_features_p, input_num_list_p, input_bbox_p = MODEL.inputs_placeholder(input_channels=1, batch_size=1)
is_training_p = tf.placeholder(dtype=tf.bool, shape=[], name='is_training')


anchors, proposals, conf = \
    MODEL.model(input_coors=input_coors_p,
                input_features=input_features_p,
                input_num_list=input_num_list_p,
                is_training=is_training_p,
                trainable=False,
                mem_saving=False,
                bn=1.)

init_op = tf.initialize_all_variables()
saver = tf.train.Saver()
TF_CONFIG = tf.ConfigProto()
TF_CONFIG.gpu_options.visible_device_list = "0"
TF_CONFIG.gpu_options.allow_growth = True
TF_CONFIG.allow_soft_placement = False
TF_CONFIG.log_device_placement = False


if __name__ == '__main__':
    with tf.Session(config=TF_CONFIG) as sess:
        sess.run(init_op)
        # saver.restore(sess, model_path)
        prediction_output = []
        overall_iou = []
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        for frame_id in tqdm(range(len(input_coors_stack))):
            batch_input_coors = input_coors_stack[frame_id]
            batch_input_features = input_features_stack[frame_id]
            batch_input_num_list = input_num_list_stack[frame_id]
            batch_input_bboxes = input_bboxes_stack[frame_id]
            # , output_bboxes, output_coors, output_conf, output_idx = \
            #     sess.run([coors, roi_attrs, roi_coors, roi_conf, nms_idx],
            output_anchors, output_bboxes, output_conf = \
                sess.run([anchors, proposals, conf],
                         feed_dict={input_coors_p: batch_input_coors,
                                    input_features_p: batch_input_features,
                                    input_num_list_p: batch_input_num_list,
                                    input_bbox_p: batch_input_bboxes,
                                    is_training_p: False})
                         # options=run_options,
                         # run_metadata=run_metadata)

            # tl = timeline.Timeline(run_metadata.step_stats)
            # ctf = tl.generate_chrome_trace_format()
            # with open('timeline-stage1.json', 'w') as f:
            #     f.write(ctf)

        #     output_idx = output_conf > 0.3
        #     output_bboxes = output_bboxes[output_idx]
        #     output_conf = output_conf[output_idx]
        #     anchor_coors = output_anchors[0, output_idx, 3:6]
        #
        #     input_rgbs = np.zeros_like(batch_input_coors) + [255, 255, 255]
        #     anchor_rgbs = np.zeros_like(anchor_coors) + [255, 0, 0]
        #     output_rgbs = get_points_rgb_with_prob(output_conf)
        #     plot_coors = np.concatenate([batch_input_coors, anchor_coors], axis=0)
        #     plot_rgbs = np.concatenate([input_rgbs, anchor_rgbs], axis=0)
        #
        #     w = output_bboxes[:, 0]
        #     l = output_bboxes[:, 1]
        #     h = output_bboxes[:, 2]
        #     x = output_bboxes[:, 3]
        #     y = output_bboxes[:, 4]
        #     z = output_bboxes[:, 5]
        #     r = output_bboxes[:, 6]
        #
        #     c = np.zeros(len(w))
        #     d = np.zeros(len(w))
        #     pred_bboxes = np.stack([w, l, h, x, y, z, r, c, d], axis=-1)
        #     pred_bboxes = np.concatenate([pred_bboxes, np.expand_dims(output_conf, axis=-1)], axis=-1)
        #     prediction_output.append(pred_bboxes)
        #
        #     output_bboxes = input_bboxes_stack[frame_id][0]
        #     output_bboxes = output_bboxes[output_bboxes[:, 0] != 0, :]
        #     w = output_bboxes[:, 0]
        #     l = output_bboxes[:, 1]
        #     h = output_bboxes[:, 2]
        #     x = output_bboxes[:, 3]
        #     y = output_bboxes[:, 4]
        #     z = output_bboxes[:, 5]
        #     r = output_bboxes[:, 6]
        #     c = np.zeros(len(w))
        #     d = np.zeros(len(w))
        #     p = np.ones(len(w))
        #     label_bboxes = np.stack([w, l, h, x, y, z, r, c, d, p], axis=-1)
        #
        #     if visualization:
        #         # pred_bbox_params = convert_threejs_bbox_with_prob(pred_bboxes, color=output_conf) if len(pred_bboxes) > 0 else []
        #         pred_bbox_params = convert_threejs_bbox_with_prob(pred_bboxes, color=output_conf) if len(pred_bboxes) > 0 else []
        #         label_bbox_params = convert_threejs_bbox_with_colors(label_bboxes, color='red') if len(label_bboxes) > 0 else []
        #         task_name = "ID_%06d_%03d" % (frame_id, len(pred_bboxes))
        #
        #         Converter.compile(task_name=task_name,
        #                           coors=convert_threejs_coors(plot_coors),
        #                           default_rgb=plot_rgbs,
        #                           bbox_params=pred_bbox_params + label_bbox_params)
        #
        # print("Overall IoU={}".format(np.mean(overall_iou)))
