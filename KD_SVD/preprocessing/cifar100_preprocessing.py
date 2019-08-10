# coding: utf-8
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities for preprocessing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
from PIL import Image


def random_rotate(x, deg):
    theta = np.pi / 180 * tf.random_uniform([1], -deg, deg)
    x = tf.contrib.image.rotate(x, theta, interpolation="NEAREST")
    return x


def preprocess_image(image, is_training):
    with tf.variable_scope('preprocessing'):
        """
        0：双线性差值。1：最近邻居法。2：双三次插值法。3：面积插值法。
        """
        # image = tf.image.resize_images(image, (16, 16), method=1)
        # image = tf.image.resize_images(image, (32, 32), method=0)

        image = tf.to_float(image)
        # print("image_shape: ", image.get_shape().as_list())
        image = (image - np.array(
            [112.4776, 124.1058, 129.3773], dtype=np.float32).reshape(
                1, 1, 3)) / np.array([70.4587, 65.4312, 68.2094],
                                     dtype=np.float32).reshape(1, 1, 3)
        # """
        if is_training:
            image = tf.image.random_flip_left_right(image)
            image = random_rotate(image, 15)
            image = tf.pad(image, [[4, 4], [4, 4], [0, 0]])
            image = tf.random_crop(image, [32, 32, 3])
            # image = tf.random_crop(image, [16, 16, 3])

            tf.summary.image('aug_img', tf.expand_dims(image, 0))
        # else:
        # image = tf.image.resize_images(image, (16, 16), method=1)
        # image = tf.image.resize_images(image, (32, 32), method=1)

        return image
