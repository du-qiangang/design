from __future__ import print_function

import os
from subprocess import Popen, PIPE
import numpy as np
import mycuda as cuda
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
import random
import re
import math
from six import iteritems
import nn
import tensorflow as tf

class Image:
    def __init__(self, _name, _image_paths):
        self.name = _name
        self.image_paths = _image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def to_rgb(img):
    w, h = img.shape
    img_rgb = np.empty((w, h, 3), dtype=np.uint8)
    img_rgb[:, :, 0] = img_rgb[:, :, 1] = img_rgb[:, :, 2] = img
    return img_rgb


def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                if par[1]=='-':
                    lr = -1
                else:
                    lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate


def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))


def store_revision_info(src_path, output_dir, arg_string):
    try:
        # Get git hash
        cmd = ['git', 'rev-parse', 'HEAD']
        gitproc = Popen(cmd, stdout=PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_hash = stdout.strip()
    except OSError as e:
        git_hash = ' '.join(cmd) + ': ' + e.strerror

    try:
        # Get local changes
        cmd = ['git', 'diff', 'HEAD']
        gitproc = Popen(cmd, stdout=PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_diff = stdout.strip()
    except OSError as e:
        git_diff = ' '.join(cmd) + ': ' + e.strerror

    # Store a text file in the log directory
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        text_file.write('arguments: %s\n--------------------\n' % arg_string)
        text_file.write('git hash: %s\n--------------------\n' % git_hash)
        text_file.write('%s' % git_diff)


def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp)
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(Image(class_name, image_paths))

    return dataset


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths


def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    #with tf.variable_scope('triplet_loss'):
    pos_dist = cuda.reduce_sum(cuda.square(cuda.subtract(anchor, positive)), 1)
    neg_dist = cuda.reduce_sum(cuda.square(cuda.subtract(anchor, negative)), 1)

    basic_loss = cuda.add(cuda.subtract(pos_dist, neg_dist), alpha)
    loss = cuda.reduce_mean(cuda.maximum(basic_loss, 0.0), 0)

    return loss


def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars,
          log_histograms=True):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    #with tf.control_dependencies([loss_averages_op]):
    # if optimizer == 'ADAGRAD':
    #     opt = AdagradOptimizer(learning_rate)
    # elif optimizer == 'ADADELTA':
    #     opt = AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
    # elif optimizer == 'ADAM':
    #     opt = AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
    # elif optimizer == 'RMSPROP':
    #     opt = RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
    # elif optimizer == 'MOM':
    #     opt = MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    # else:
    #     raise ValueError('Invalid optimization algorithm')
    #
    # grads = opt.compute_gradients(total_loss, update_gradient_vars)
    #
    # # Apply gradients.
    # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    # loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer == 'ADAM':
            opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer == 'RMSPROP':
            opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')

        grads = opt.compute_gradients(total_loss, update_gradient_vars)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    # Add histograms for trainable variables.
    #
    #添加记录日志
    # if log_histograms:
    #     for var in tf.trainable_variables():
    #         tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    # if log_histograms:
    #     for grad, var in grads:
    #         if grad is not None:
    #             tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables)

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # return train_op
    return train_op


# class ExponentialMovingAverage:
#     def __init__(self,decay,_num_updates=0):
#         self.decay=decay
#         self.num_updates=_num_updates
#         self.shadow=[]
#
#     def apply(self,_step=None):
#         if _step is not None:
#             self.num_updates=_step
#         self.decay=np.min(self.decay,(1+self.num_updates)/(10+self.num_updates))
#
#     def average(self,vars):
#         for var in vars:
#             self.shadow.append(self.decay*self.shadow+(1-self.decay)*var)
#         return self.shadow
#
#
# class AdagradOptimizer:
#     def __init__(self,learning_rate,use_nesterov=False):
#         self.learning_rate=learning_rate
#         self.use_nesterov=use_nesterov
#
#     def compute_gradients(self,loss, val_list):
#         gradient=[]
#
#         return gradient,val_list
#
#     def apply_gradients(self,grads_and_vars, global_step=None):
#         return
#
#
# class AdadeltaOptimizer:
#     def __init__(self,learning_rate,momentum,use_nesterov=False):
#         self.learning_rate=learning_rate
#         self.momentum=momentum
#         self.use_nesterov=use_nesterov
#
#     def compute_gradients(self,loss, val_list):
#         gradient=[]
#
#         return gradient,val_list
#
#     def apply_gradients(self,grads_and_vars, global_step=None):
#         return
#
#
# class AdamOptimizer:
#     def __init__(self,learning_rate=0.001,
#                beta1=0.9, beta2=0.999, epsilon=1e-8,
#                use_locking=False,):
#         self.lr=learning_rate
#         self._beta1 = beta1
#         self._beta2 = beta2
#         self._epsilon = epsilon
#
#         self._lr_t = None
#         self._beta1_t = None
#         self._beta2_t = None
#         self._epsilon_t = None
#
#     def compute_gradients(self,loss, val_list,aggregation_method=None,colocate_gradients_with_ops=False,
#         grad_loss=None):
#         gradient=[]
#
#         return gradient,val_list
#
#     def apply_gradients(self,grads_and_vars, global_step=None):
#         return
#
#     def _creat(self,vars):
#         self.v=np.zeros_like(vars)
#         self.m=np.zeros_like(vars)
#
#     def val_update(self,grad,vars):
#         t=0
#         while True:
#             t=t+1
#             self.lr=self.lr*np.sqrt(1-self._beta2**t)/(1-self._beta1**t)
#             self.m=self._beta1*self.m+(1-self._beta1)*grad
#             self.v=self._beta2*self.v+(1-self._beta2)*grad*grad
#             vars -= (self.lr*self.m)/(np.sqrt(self.v)+self._epsilon)
#
# class RMSPropOptimizer:
#     def __init__(self,learning_rate,momentum,use_nesterov=False):
#         self.learning_rate=learning_rate
#         self.momentum=momentum
#         self.use_nesterov=use_nesterov
#
#     def compute_gradients(self,loss, val_list):
#         gradient=[]
#
#         return gradient,val_list
#
#     def apply_gradients(self,grads_and_vars, global_step=None):
#         return
#
#
# class MomentumOptimizer:
#     def __init__(self,learning_rate,momentum,use_nesterov=False):
#         self.learning_rate=learning_rate
#         self.momentum=momentum
#         self.use_nesterov=use_nesterov
#
#     def compute_gradients(self,loss, val_list):
#         gradient=[]
#
#         return gradient,val_list
#
#     def apply_gradients(self,grads_and_vars, global_step=None):
#         return


def _add_loss_summaries(total_loss):
    """Add summaries for losses.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9)
    losses = nn.get_collection('losses')

    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    #for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # # as the original loss name.
        # tf.summary.scalar(l.op.name + ' (raw)', l)
        # tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op