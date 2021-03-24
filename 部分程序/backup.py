from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pycuda.tools
import pycuda.autoinit
from datetime import datetime
import os.path
import time
import sys
from facenet.src import mycuda as cuda
from facenet.src import nn
import numpy as np
import importlib
import itertools
import argparse
from facenet.src import cfacenet
import facenet.src.lfw as lfw
from six.moves import xrange
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from queue import Queue
import pandas
import cv2
import random

dev_batch = None


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.', default='~/models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        default='/home/tangruiqi/dqg_cuda/python/facenet/src/lfw_160')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--people_per_batch', type=int,
                        help='Number of people per batch.', default=45)
    parser.add_argument('--images_per_person', type=int,
                        help='Number of images per person.', default=40)
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--alpha', type=float,
                        help='Positive to negative triplet distance margin.', default=0.2)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop',
                        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
                             'If the size of the images in the data directory is equal to image_size no cropping is performed',
                        action='store_true')
    parser.add_argument('--random_flip',
                        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--learning_rate_schedule_file', type=str,
                        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.',
                        default='data/learning_rate_schedule.txt')

    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_dir', type=str,
                        help='Path to the data directory containing aligned face patches.', default='')
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)


def train(args, dataset, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, input_queue,
          global_step,
          embeddings, loss, train_op, summary_op, summary_writer, learning_rate_schedule_file,
          embedding_size, anchor, positive, negative, triplet_loss):
    batch_number = 0

    if args.learning_rate > 0.0:
        lr = args.learning_rate
    else:
        lr = cfacenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    while batch_number < args.epoch_size:
        # 从数据集获取数据并随机打乱顺序
        image_paths, num_per_class = sample_people(dataset, args.people_per_batch, args.images_per_person)

        print('Running forward pass on sampled images: ', end='')
        start_time = time.time()
        nrof_examples = args.people_per_batch * args.images_per_person  # 数据总量
        labels_array = np.reshape(np.arange(nrof_examples), (-1, 3))
        image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))
        # 将数据入队
        # sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
        enqueue_op.put(image_paths_array, labels_array)
        emb_array = np.zeros((nrof_examples, embedding_size))
        #   根据batch大小划分的batch总量
        nrof_batches = int(np.ceil(nrof_examples / args.batch_size))
        #   读入embedding，如果数据不足一个batch，以小量作为batch大小
        for i in range(nrof_batches):
            batch_size = min(nrof_examples - i * args.batch_size, args.batch_size)
            phase_train = True
            emb = batch_size
            lab = lr
            emb_array[lab, :] = emb
        print('%.3f' % (time.time() - start_time))

        # 根据embedding选择三元组
        print('Selecting suitable triplets for training')
        triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array, num_per_class,
                                                                    image_paths, args.people_per_batch, args.alpha)
        selection_time = time.time() - start_time
        print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' %
              (nrof_random_negs, nrof_triplets, selection_time))

        # 根据选择的三元组进行训练
        nrof_batches = int(np.ceil(nrof_triplets * 3 / args.batch_size))
        triplet_paths = list(itertools.chain(*triplets))
        labels_array = np.reshape(np.arange(len(triplet_paths)), (-1, 3))
        triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths), 1), (-1, 3))
        #   将三元组添加到数据集
        enqueue_op.put(triplet_paths_array, labels_array)
        nrof_examples = len(triplet_paths)
        train_time = 0
        i = 0
        emb_array = np.zeros((nrof_examples, embedding_size))
        loss_array = np.zeros((nrof_triplets,))
        # summary = tf.Summary()
        step = 0
        while i < nrof_batches:
            start_time = time.time()
            batch_size = min(nrof_examples - i * args.batch_size, args.batch_size)
            # feed_dict = {batch_size_placeholder: batch_size, learning_rate_placeholder: lr,
            #              phase_train_placeholder: True}
            # err, _, step, emb, lab = sess.run([loss, train_op, global_step, embeddings, labels_batch],
            #                                   feed_dict=feed_dict)
            # 进行下列四个操作函数
            err = loss
            step = global_step
            emb = embeddings
            lab = labels_batch
            emb_array[lab, :] = emb
            loss_array[i] = err
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
                  (epoch, batch_number + 1, args.epoch_size, duration, err))
            batch_number += 1
            i += 1
            train_time += duration
            # summary.value.add(tag='loss', simple_value=err)

        # Add validation loss and accuracy to summary
        # pylint: disable=maybe-no-member
        # summary.value.add(tag='time/selection', simple_value=selection_time)
        # summary_writer.add_summary(summary, step)
    return step


def sample_people(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person

    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths) < nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images - len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index] * nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i += 1

    return image_paths, num_per_class


def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []

    for i in xrange(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in xrange(1, nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in xrange(j, nrof_images):  # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx + nrof_images] = np.NaN
                all_neg = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs > 0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    # summary = tf.Summary()
    # pylint: disable=maybe-no-member
    # summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    # summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    # summary_writer.add_summary(summary, step)


def evaluate(image_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder,
             batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, actual_issame,
             batch_size,
             nrof_folds, log_dir, step, summary_writer, embedding_size):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Running forward pass on LFW images: ', end='')

    nrof_images = len(actual_issame) * 2
    assert (len(image_paths) == nrof_images)
    labels_array = np.reshape(np.arange(nrof_images), (-1, 3))
    image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))
    # sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    enqueue_op.put(image_paths_array, labels_array)
    emb_array = np.zeros((nrof_images, embedding_size))
    nrof_batches = int(np.ceil(nrof_images / batch_size))
    label_check_array = np.zeros((nrof_images,))
    for i in xrange(nrof_batches):
        batch_size = min(nrof_images - i * batch_size, batch_size)
        emb = embeddings
        lab = labels_batch(batch_size, learning_rate_placeholder=0.0)
        emb_array[lab, :] = emb
        label_check_array[lab] = 1
    print('%.3f' % (time.time() - start_time))

    assert (np.all(label_check_array == 1))

    _, _, accuracy, val, val_std, far = lfw.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)

    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    # summary = tf.Summary()
    # pylint: disable=maybe-no-member
    # summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    # summary.value.add(tag='lfw/val_rate', simple_value=val)
    # summary.value.add(tag='time/lfw', simple_value=lfw_time)
    # summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir, 'lfw_result.txt'), 'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))


def main(args):
    network = importlib.import_module(args.model_def)

    # 建立日志目录和模型目录
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # 向日志目录中保存参数信息
    cfacenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))

    # 保存git相关的信息
    # src_path, _ = os.path.split(os.path.realpath(__file__))
    # cfacenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    # 获取整体数据集
    np.random.seed(seed=args.seed)
    train_set = cfacenet.get_dataset(args.data_dir)

    # 预训练模型路径
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    if args.pretrained_model:
        print('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))

    # lfw数据集路径
    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)

    # with tf.Graph().as_default():
    # tf.set_random_seed(args.seed)
    random.seed(args.seed)
    global_step = 0

    # 占位符
    learning_rate = np.zeros((1,), np.float32)
    #
    batch_size = np.zeros((1,), np.int32)
    #
    phase_train = np.array([True, ], dtype=np.bool)
    #
    image_paths = np.zeros(shape=(None, 3), dtype="", )
    labels = np.zeros(shape=(None, 3), dtype=np.int64, )

    input_queue = Queue(maxsize=100000)
    enqueue_op = input_queue.put([image_paths, labels])

    nrof_preprocess_threads = 4
    images_and_labels = []
    # 读入图片，并进行翻转裁剪等处理以及标签读入
    for _ in range(nrof_preprocess_threads):
        filenames, label = input_queue.get()

        images = []
        for filename in pandas.DataFrame(filenames).unstack():
            image = cv2.imread(filename, cv2.IMREAD_COLOR)

            if args.random_crop:
                # 随机裁剪
                image = cuda.random_crop(image, [args.image_size, args.image_size, 3])
            else:
                # 裁剪或填充
                image = cuda.resize_image_with_crop_or_pad(image, args.image_size)
            if args.random_flip:
                # 概率左右翻转
                image = cuda.random_flip_left_right(image)

                # pylint: disable=no-member
            image.set_shape((args.image_size, args.image_size, 3))
            images.append(cuda.per_image_standardization(image))
        images_and_labels.append([images, label])
    # 多线程在数据集中将图像和标签分开
    image_batch, labels_batch = nn.batch_join(
        images_and_labels, batch_size=batch_size,
        shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
        capacity=4 * nrof_preprocess_threads * args.batch_size,
        allow_smaller_final_batch=True)
    # image_batch = tf.identity(image_batch, 'image_batch')
    # image_batch = tf.identity(image_batch, 'input')
    # labels_batch = tf.identity(labels_batch, 'label_batch')

    #
    prelogits, _ = network.inference(image_batch, args.keep_probability,
                                     phase_train=phase_train, bottleneck_layer_size=args.embedding_size,
                                     weight_decay=args.weight_decay)

    embeddings = nn.operator.l2_normalize(prelogits, 1, 1e-10)
    # 获取triplet输入数据embedding，并计算损失函数
    anchor, positive, negative = nn.unstack(embeddings.reshape([-1, 3, args.embedding_size]), 3, 1)
    triplet_loss = cfacenet.triplet_loss(anchor, positive, negative, args.alpha)

    learning_rate = cuda.exponential_decay(learning_rate, global_step,
                                           args.learning_rate_decay_epochs * args.epoch_size,
                                           args.learning_rate_decay_factor, staircase=True)
    # tf.summary.scalar('learning_rate', learning_rate)

    # 获取 total losses
    # regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')
    total_loss = np.sum(triplet_loss)
    # 建立一个图，训练并更新参数
    train_op = cfacenet.train(total_loss, global_step, args.optimizer,
                              learning_rate, args.moving_average_decay)

    # Create a saver
    # saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

    # Build the summary operation based on the TF collection of Summaries.
    # summary_op = tf.summary.merge_all()

    # Initialize variables
    # sess.run(tf.global_variables_initializer(), feed_dict={phase_train: True})
    # sess.run(tf.local_variables_initializer(), feed_dict={phase_train: True})

    # summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    # coord = tf.train.Coordinator()
    # tf.train.start_queue_runners(coord=coord, sess=sess)

    # 导入预训练模型
    if args.pretrained_model:
        print('Restoring pretrained model: %s' % args.pretrained_model)
        # saver.restore(sess, os.path.expanduser(args.pretrained_model))

        # Training and validation loop
    epoch = 0
    while epoch < args.max_nrof_epochs:
        # step = sess.run(global_step, feed_dict=None)
        step = 0
        epoch = step // args.epoch_size
        # 用一个epoch的数据训练
        train(args, train_set, epoch, image_paths, labels, labels_batch,
              batch_size, learning_rate, phase_train, enqueue_op,
              input_queue, global_step,
              embeddings, total_loss, train_op, args.learning_rate_schedule_file,
              args.embedding_size, anchor, positive, negative, triplet_loss)

        # 保存模型
        # save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

        # 验证模型性能
        if args.lfw_dir:
            evaluate(lfw_paths, embeddings, labels_batch, image_paths, labels,
                     batch_size, learning_rate, phase_train, enqueue_op,
                     actual_issame, args.batch_size,
                     args.lfw_nrof_folds, log_dir, step, args.embedding_size)

    return model_dir


model = SourceModule("""
__global__ void train(float *embeddings){

}
__global__ void embedding(x,dim,epsilno=1e-12){

}
""")

# def train_gpu(args):
#     func=model.get_function("train")
#     network = importlib.import_module(args.model_def)
#
#     # 建立日志目录和模型目录
#     subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
#     log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
#     if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
#         os.makedirs(log_dir)
#     model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
#     if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
#         os.makedirs(model_dir)
#
#     # 向日志目录中保存参数信息
#     cfacenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
#
#     # 保存git相关的信息
#     src_path, _ = os.path.split(os.path.realpath(__file__))
#     cfacenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))
#
#     # 获取整体数据集
#     np.random.seed(seed=args.seed)
#     train_set = cfacenet.get_dataset(args.data_dir)
#
#     # 预训练模型路径
#     print('Model directory: %s' % model_dir)
#     print('Log directory: %s' % log_dir)
#     if args.pretrained_model:
#         print('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))
#
#     # lfw数据集路径
#     if args.lfw_dir:
#         print('LFW directory: %s' % args.lfw_dir)
#         # Read the file containing the pairs used for testing
#         pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
#         # Get the paths for the corresponding images
#         lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)
#
#     random.seed(args.seed)
#     global_step = 0
#     learning_rate = np.float32(0)
#     batch_size = np.int32(0)
#     phase_train = np.bool(True)
#     image_paths = np.zeros(shape=(None, 3), dtype="", )
#     labels = np.zeros(shape=(None, 3), dtype=np.int64, )
#
#     input_queue = Queue(maxsize=100000)
#     enqueue_op = input_queue.put([image_paths, labels])
#
#     nrof_preprocess_threads = 4
#     images_and_labels = []
#     # 读入图片，并进行翻转裁剪等处理以及标签读入
#     for _ in range(nrof_preprocess_threads):
#         filenames, label = input_queue.get()
#
#         images = []
#         for filename in pandas.DataFrame(filenames).unstack():
#             image = cv2.imread(filename, cv2.IMREAD_COLOR)
#
#             if args.random_crop:
#                 # 随机裁剪
#                 image = cuda.random_crop(image, [args.image_size, args.image_size, 3])
#             else:
#                 # 裁剪或填充
#                 image = cuda.resize_image_with_crop_or_pad(image, args.image_size)
#             if args.random_flip:
#                 # 概率左右翻转
#                 image = cuda.random_flip_left_right(image)
#
#                 # pylint: disable=no-member
#             image.set_shape((args.image_size, args.image_size, 3))
#             images.append(cuda.per_image_standardization(image))
#         images_and_labels.append([images, label])
#
#     image_batch, labels_batch = tf.train.batch_join(
#         images_and_labels, batch_size=batch_size,
#         shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
#         capacity=4 * nrof_preprocess_threads * args.batch_size,
#         allow_smaller_final_batch=True)
#     # image_batch = tf.identity(image_batch, 'image_batch')
#     # image_batch = tf.identity(image_batch, 'input')
#     # labels_batch = tf.identity(labels_batch, 'label_batch')
#
#     #
#     prelogits, _ = network.inference(image_batch, args.keep_probability,
#                                      phase_train=phase_train, bottleneck_layer_size=args.embedding_size,
#                                      weight_decay=args.weight_decay)
#
#     embeddings = nn.l2_normalize(prelogits, 1, 1e-10)
#     # 获取triplet输入数据embedding，并计算损失函数
#     anchor, positive, negative = np.unstack(embeddings.reshape([-1, 3, args.embedding_size]), 3, 1)
#     triplet_loss = cfacenet.triplet_loss(anchor, positive, negative, args.alpha)
#
#     learning_rate = cuda.exponential_decay(learning_rate, global_step,
#                                            args.learning_rate_decay_epochs * args.epoch_size,
#                                            args.learning_rate_decay_factor, staircase=True)
#     # tf.summary.scalar('learning_rate', learning_rate)
#
#     # 获取 total losses
#     regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#     total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')
#
#     # 建立一个图，训练并更新参数
#     train_op = cfacenet.train(total_loss, global_step, args.optimizer,
#                               learning_rate, args.moving_average_decay, tf.global_variables())
#
#     # Create a saver
#     # saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
#
#     # Build the summary operation based on the TF collection of Summaries.
#     # summary_op = tf.summary.merge_all()
#
#     # Initialize variables
#     sess.run(tf.global_variables_initializer(), feed_dict={phase_train: True})
#     sess.run(tf.local_variables_initializer(), feed_dict={phase_train: True})
#
#     # summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
#     # coord = tf.train.Coordinator()
#     # tf.train.start_queue_runners(coord=coord, sess=sess)
#
#     # 导入预训练模型
#     if args.pretrained_model:
#         print('Restoring pretrained model: %s' % args.pretrained_model)
#         saver.restore(sess, os.path.expanduser(args.pretrained_model))
#
#         # Training and validation loop
#     epoch = 0
#     while epoch < args.max_nrof_epochs:
#         #取得global_step
#         step = global_step
#         epoch = step // args.epoch_size
#         # 用一个epoch的数据训练
#         train(args, sess, train_set, epoch, image_paths, labels, labels_batch,
#               batch_size, learning_rate, phase_train, enqueue_op,
#               input_queue, global_step,
#               embeddings, total_loss, train_op, summary_op, summary_writer, args.learning_rate_schedule_file,
#               args.embedding_size, anchor, positive, negative, triplet_loss)
#
#         # 保存模型
#         save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)
#
#         # 验证模型性能
#         if args.lfw_dir:
#             evaluate(sess, lfw_paths, embeddings, labels_batch, image_paths, labels,
#                      batch_size, learning_rate, phase_train, enqueue_op,
#                      actual_issame, args.batch_size,
#                      args.lfw_nrof_folds, log_dir, step, summary_writer, args.embedding_size)
#
#     return model_dir
#
