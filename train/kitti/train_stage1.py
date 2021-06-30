import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import horovod.tensorflow as hvd
import os
from tqdm import tqdm
from os.path import join, dirname
import sys
import argparse
from shutil import rmtree, copyfile

HOME = join(dirname(os.getcwd()))
sys.path.append(HOME)

from models.builder.kitti import model_stage1 as MODEL
from configs.kitti import kitti_config_training as CONFIG
from data.generator.kitti_generator import KittiDataset
from train.train_utils import get_train_op, get_config, save_best_sess, set_training_controls

hvd.init()
is_hvd_root = hvd.rank() == 0

if CONFIG.local:
    log_dir = '/home/tan/tony/dv-det-v1.0/checkpoints/local-debug'
    try: rmtree(log_dir)
    except: pass
    os.mkdir(log_dir)
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', dest='log_dir', default='test')
    args = parser.parse_args()
    log_dir = args.log_dir

if is_hvd_root:
    copyfile(CONFIG.config_dir, join(log_dir, CONFIG.config_dir.split('/')[-1]))

DatasetTrain = KittiDataset(task="training",
                            batch_size=CONFIG.batch_size_stage1,
                            config=CONFIG.aug_config,
                            num_worker=CONFIG.num_worker,
                            hvd_size=hvd.size(),
                            hvd_id=hvd.rank())

DatasetValid = KittiDataset(task="validation",
                            validation=True,
                            batch_size=CONFIG.batch_size_stage1,
                            hvd_size=hvd.size(),
                            hvd_id=hvd.rank())

training_batch = DatasetTrain.batch_sum
validation_batch = DatasetValid.batch_sum
decay_batch = training_batch * CONFIG.decay_epochs

input_coors_p, input_features_p, input_num_list_p, input_bbox_p = \
    MODEL.inputs_placeholder(input_channels=1,
                             bbox_padding_num=CONFIG.bbox_padding)
is_training_p = tf.placeholder(dtype=tf.bool, shape=[], name="is_stage1_training")


tf_step = tf.Variable(0, name='stage1_step')
lr, bn, wd = set_training_controls(config=CONFIG,
                                   lr=CONFIG.init_lr_stage1,
                                   scale=CONFIG.lr_scale_stage1,
                                   decay_batch=decay_batch,
                                   lr_warm_up=CONFIG.lr_warm_up,
                                   step=tf_step,
                                   hvd_size=hvd.size(),
                                   prefix='stage1')
loader = tf.train.Saver()

anchors, proposals, pred_conf = \
    MODEL.model(input_coors=input_coors_p,
                input_features=input_features_p,
                input_num_list=input_num_list_p,
                is_training=is_training_p,
                trainable=True,
                mem_saving=True,
                bn=bn)

tf_loss, tf_iou = MODEL.loss(anchors=anchors,
                             proposals=proposals,
                             pred_conf=pred_conf,
                             labels=input_bbox_p,
                             weight_decay=wd)

train_op = get_train_op(tf_loss, lr, var_keywords=['stage1'], opt='adam', global_step=tf_step, use_hvd=True)

tf_summary = tf.summary.merge_all()
hooks = [hvd.BroadcastGlobalVariablesHook(0)]
session_config = get_config(gpu=CONFIG.gpu_list[hvd.rank()])

training_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'))
validation_writer = tf.summary.FileWriter(os.path.join(log_dir, 'valid'))
saver = tf.train.Saver(max_to_keep=None)



def train_one_epoch(sess, step, dataset_generator, writer):
    iou_sum = 0
    iter = tqdm(range(training_batch)) if is_hvd_root else range(training_batch)
    for _ in iter:
        coors, features, num_list, bboxes = next(dataset_generator)
        iou, _, summary = sess.run([tf_iou, train_op, tf_summary],
                                    feed_dict={input_coors_p: coors,
                                               input_features_p: features,
                                               input_num_list_p: num_list,
                                               input_bbox_p: bboxes,
                                               is_training_p: True})

        iou_sum += iou
        step += 1
        if is_hvd_root:
            writer.add_summary(summary, step)

    iou = iou_sum / training_batch

    if is_hvd_root:
        summary = tf.Summary()
        summary.value.add(tag='stage1_overall_IoU', simple_value=iou)
        writer.add_summary(summary, step)
        print("Training: Total IoU={:0.2f}".format(iou))
    return step


def valid_one_epoch(sess, step, dataset_generator, writer):
    iou_sum = 0
    instance_count = 0
    iter = tqdm(range(validation_batch)) if is_hvd_root else range(validation_batch)
    for _, batch_id in enumerate(iter):
        coors, features, num_list, bboxes = next(dataset_generator)
        iou, summary = sess.run([tf_iou, tf_summary],
                                feed_dict={input_coors_p: coors,
                                           input_features_p: features,
                                           input_num_list_p: num_list,
                                           input_bbox_p: bboxes,
                                           is_training_p: False})

        iou_sum += (iou * len(features))
        instance_count += len(features)
        if is_hvd_root:
            writer.add_summary(summary, step + batch_id)

    iou = iou_sum / instance_count

    if is_hvd_root:
        summary = tf.Summary()
        summary.value.add(tag='stage1_overall_IoU', simple_value=iou)
        writer.add_summary(summary, step)
        print("Validation: Total IoU={:0.2f}".format(iou))
    return iou


def main():
    with tf.train.MonitoredTrainingSession(hooks=hooks, config=session_config) as mon_sess:

        train_generator = DatasetTrain.train_generator()
        valid_generator = DatasetValid.valid_generator()
        best_result = 0.
        step = 0

        for epoch in range(CONFIG.total_epoch):
            if is_hvd_root:
                print("Epoch: {}".format(epoch))

            step = train_one_epoch(sess=mon_sess,
                                   step=step,
                                   dataset_generator=train_generator,
                                   writer=training_writer)

            if epoch % CONFIG.valid_interval == 0:  # and EPOCH != 0:
                result = valid_one_epoch(sess=mon_sess,
                                         step=step,
                                         dataset_generator=valid_generator,
                                         writer=validation_writer)
                if is_hvd_root:
                    best_result = save_best_sess(sess=mon_sess,
                                                 best_acc=best_result,
                                                 acc=result,
                                                 log_dir=log_dir,
                                                 saver=saver,
                                                 replace=True,
                                                 log=is_hvd_root,
                                                 inverse=False,
                                                 save_anyway=False)

if __name__ == '__main__':
    main()
