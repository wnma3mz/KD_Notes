# coding: utf-8
import os

import numpy as np
import scipy.io as sio
import tensorflow as tf
from tensorflow import ConfigProto
from tensorflow.contrib import slim
from tensorflow.python.ops import control_flow_ops as cfo

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from utils import distillation_learning_rate, GET_dataset, sigmoid, _get_init_fn, MODEL

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

# train_dir   =  '/home/dmsl/Documents/tf/svd/VGG/VGG'
# dataset_dir = '/home/dmsl/Documents/data/tf/cifar100'

train_dir = 'output_origin_teacher'
dataset_dir = 'cifar-100-python'
dataset_name = 'cifar100'
model_name = 'VGG_teacher'
preprocessing_name = 'cifar100'

Optimizer = 'sgd'  # 'adam' or 'sgd'
Learning_rate = 1e-2

batch_size = 128
val_batch_size = 200
init_epoch = 0
num_epoch = 200 + init_epoch
weight_decay = 1e-4

checkpoint_path = None
# checkpoint_path   =  '/home/dmsl/Documents/tf/svd/mobile/mobile'
ignore_missing_vars = True
### main

tf.logging.set_verbosity(tf.logging.INFO)

with tf.Graph().as_default() as graph:
    # Load Dataset
    epoch, global_step, dataset, image, label = GET_dataset(
        dataset_name, dataset_dir, batch_size, preprocessing_name, 'train')

    _, _, val_dataset, val_image, val_label = GET_dataset(
        dataset_name, dataset_dir, val_batch_size, preprocessing_name, 'test')

    with tf.device('/device:CPU:0'):
        decay_steps = dataset.num_samples // batch_size
        max_number_of_steps = int(dataset.num_samples / batch_size *
                                  (num_epoch))

    total_loss, train_accuracy = MODEL(model_name, weight_decay, image, label,
                                       Learning_rate, epoch, True)

    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    dlr = distillation_learning_rate(Learning_rate, epoch, init_epoch)
    optimizer = tf.train.MomentumOptimizer(dlr, 0.9, use_nesterov=True)

    gradient0 = optimizer.compute_gradients(total_loss[0], var_list=variables)

    op = optimizer.apply_gradients(gradient0, global_step=global_step)
    update_ops.append(op)
    update_op = tf.group(*update_ops)
    train_op = cfo.with_dependencies([update_op],
                                     tf.add_n(total_loss),
                                     name='train_op')

    val_loss, val_accuracy = MODEL(model_name, weight_decay, val_image,
                                   val_label, Learning_rate, epoch, False,
                                   True)

    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summary_op = tf.summary.merge(list(summaries), name='summary_op')

    # for validation
    def ts_fn(session, *args, **kwargs):
        total_loss, should_stop = slim.learning.train_step(
            session, *args, **kwargs)
        if (ts_fn.step % (ts_fn.decay_steps) == 0):
            accuracy = 0
            itr = val_dataset.num_samples // val_batch_size
            for _ in range(itr):
                accuracy += session.run(ts_fn.val_accuracy)
            print(
                'Epoch %s Step %s - Loss: %.2f Accuracy: %.2f%%, Highest Accuracy : %.2f%%'
                %
                (str((ts_fn.step - ts_fn.decay_steps * ts_fn.init_epoch) //
                     ts_fn.decay_steps).rjust(3, '0'),
                 str(ts_fn.step - ts_fn.decay_steps * ts_fn.init_epoch).rjust(
                     6, '0'), total_loss, accuracy * 100 / itr,
                 ts_fn.highest * 100 / itr))
            acc = tf.Summary(value=[
                tf.Summary.Value(tag="Accuracy",
                                 simple_value=accuracy * 100 / itr)
            ])
            ts_fn.eval_writer.add_summary(
                acc, ts_fn.step - ts_fn.decay_steps * ts_fn.init_epoch)

            var = {}
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            for v in variables:
                var[v.name[:-2]] = session.run(v)
            sio.savemat(os.path.join(train_dir, 'trained_params.mat'), var)
            print('save new parameters')
            ts_fn.highest = accuracy

        ts_fn.step += 1
        return [total_loss, should_stop]

    ts_fn.saver = tf.train.Saver()
    ts_fn.eval_writer = tf.summary.FileWriter(os.path.join(train_dir, 'eval'),
                                              graph,
                                              flush_secs=60)
    ts_fn.step = 0
    ts_fn.decay_steps = decay_steps
    ts_fn.init_epoch = init_epoch
    ts_fn.val_accuracy = val_accuracy
    ts_fn.highest = 0

    # training
    config = ConfigProto()
    config.gpu_options.allow_growth = True

    # tf.contrib.training.train(train_op, train_dir)
    slim.learning.train(
        train_op,
        logdir=train_dir,
        global_step=global_step,
        session_config=config,
        init_fn=_get_init_fn(checkpoint_path, ignore_missing_vars),
        summary_op=summary_op,
        train_step_fn=ts_fn,
        number_of_steps=max_number_of_steps,
        log_every_n_steps=40,  #'The frequency with which logs are print.'
        save_summaries_secs=
        120,  #'The frequency with which summaries are saved, in seconds.'
        save_interval_secs=0
    )  #'The frequency with which the model is saved, in seconds.'
