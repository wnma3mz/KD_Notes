# coding: utf-8
import tensorflow as tf
from preprocessing import preprocessing_factory
from nets import nets_factory
from datasets import dataset_factory
from tensorflow.contrib import slim

# 训练时用到的函数全部迁移到此py文件


def sigmoid(x, k):
    return 1 / (1 + tf.exp(-(x - k)))


def _get_init_fn(checkpoint_path, ignore_missing_vars):
    if checkpoint_path is None:
        return None
    variables_to_restore = slim.get_variables_to_restore()[1:]
    for v in variables_to_restore:
        print(v)

    if tf.gfile.IsDirectory(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=ignore_missing_vars)


# Compute Loss & Gradient
def distillation_learning_rate(Learning_rate, epoch, init_epoch):
    Learning_rate = tf.case([
        (tf.less(epoch, 100 + init_epoch), lambda: Learning_rate),
        (tf.less(epoch, 150 + init_epoch), lambda: Learning_rate * 1e-1),
    ],
                            default=lambda: Learning_rate * 1e-2)
    tf.summary.scalar('learning_rate', Learning_rate)
    return Learning_rate


def GET_dataset(dataset_name, dataset_dir, batch_size, preprocessing_name,
                split):
    if split == 'train':
        sff = True
        threads = 16
        is_training = True
    else:
        sff = False
        threads = 16
        is_training = False
    with tf.variable_scope('dataset_%s' % split):
        # 载入数据集
        dataset = dataset_factory.get_dataset(dataset_name, split, dataset_dir)
        with tf.device('/device:CPU:0'):
            if split == 'train':
                # global_step = tf.train.create_global_step()
                global_step = slim.create_global_step()
                p = tf.floor_div(
                    tf.cast(global_step, tf.float32),
                    tf.cast(int(dataset.num_samples / float(batch_size)),
                            tf.float32))
            else:
                global_step = None
                p = None
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                shuffle=sff,
                num_readers=threads,
                common_queue_capacity=dataset.num_samples,
                common_queue_min=0)
        images, labels = provider.get(['image', 'label'])
        # 预处理
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name, is_training)
        images = image_preprocessing_fn(images)
        if split == 'train':
            batch_images, batch_labels = tf.train.shuffle_batch(
                [images, labels],
                batch_size=batch_size,
                num_threads=threads,
                capacity=dataset.num_samples,
                min_after_dequeue=0)
            with tf.variable_scope('1-hot_encoding'):
                batch_labels = slim.one_hot_encoding(batch_labels,
                                                     dataset.num_classes,
                                                     on_value=1.0)

            batch_queue = slim.prefetch_queue.prefetch_queue(
                [batch_images, batch_labels], capacity=40 * batch_size)

            image, label = batch_queue.dequeue()

        else:
            batch_images, batch_labels = tf.train.batch(
                [images, labels],
                batch_size=batch_size,
                num_threads=threads,
                capacity=dataset.num_samples)

            with tf.variable_scope('1-hot_encoding'):
                batch_labels = slim.one_hot_encoding(batch_labels,
                                                     dataset.num_classes,
                                                     on_value=1.0)
            batch_queue = slim.prefetch_queue.prefetch_queue(
                [batch_images, batch_labels], capacity=8 * batch_size)

            image, label = batch_queue.dequeue()
    return p, global_step, dataset, image, label


def MODEL(model_name,
          weight_decay,
          image,
          label,
          lr,
          epoch,
          is_training,
          is_teacher=True):
    network_fn = nets_factory.get_network_fn(model_name,
                                             weight_decay=weight_decay)
    end_points = network_fn(image,
                            is_training=is_training,
                            lr=lr,
                            val=not (is_training))
    losses = []

    if is_training:

        def scale_grad(x, scale):
            return scale * x + tf.stop_gradient((1 - scale) * x)

        with tf.variable_scope('Student_loss'):
            # 计算损失和精度
            loss = tf.losses.softmax_cross_entropy(label, end_points['Logits'])
            accuracy = slim.metrics.accuracy(
                tf.to_int32(tf.argmax(end_points['Logits'], 1)),
                tf.to_int32(tf.argmax(label, 1)))
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', accuracy)
            losses.append(loss +
                          tf.add_n(tf.losses.get_regularization_losses()))

        # """
        if not is_teacher:
            with tf.variable_scope('Dist_loss'):
                dist_loss = end_points['Dist']
                tf.summary.scalar('dist_loss', dist_loss)
                losses.append(dist_loss)
        # """

    else:
        losses = tf.losses.softmax_cross_entropy(label, end_points['Logits'])
        accuracy = slim.metrics.accuracy(
            tf.to_int32(tf.argmax(end_points['Logits'], 1)),
            tf.to_int32(tf.argmax(label, 1)))

    return losses, accuracy
