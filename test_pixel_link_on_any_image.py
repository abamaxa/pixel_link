#encoding = utf-8
import time
import os
import math
import pprint
import cProfile

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.training.python.training import evaluation
from datasets import dataset_factory
from preprocessing import ssd_vgg_preprocessing
from tf_extended import metrics as tfe_metrics
import util
import cv2
import pixel_link

from nets import pixel_link_symbol
from post_process.post_process import PyPixelLinkPostProcess

import pixel_link_infer

import PIL

slim = tf.contrib.slim
import config
# =========================================================================== #
# Checkpoint and running Flags
# =========================================================================== #
tf.app.flags.DEFINE_string('checkpoint_path', None, 
   'the path of pretrained model to be used. If there are checkpoints\
    in train_dir, this config will be ignored.')

tf.app.flags.DEFINE_float('gpu_memory_fraction', -1, 
  'the gpu memory fraction to be used. If less than 0, allow_growth = True is used.')


# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_dir', 'None', 
    'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer('eval_image_width', None, 'resized image width for inference')
tf.app.flags.DEFINE_integer('eval_image_height',  None, 'resized image height for inference')
tf.app.flags.DEFINE_float('pixel_conf_threshold',  None, 'threshold on the pixel confidence')
tf.app.flags.DEFINE_float('link_conf_threshold',  None, 'threshold on the link confidence')

tf.app.flags.DEFINE_bool('using_moving_average', False, 
                         'Whether to use ExponentionalMovingAverage')
                    
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, 
                          'The decay rate of ExponentionalMovingAverage')

tf.app.flags.DEFINE_string('model_name', None, 'Name of Mobilenet model.')

FLAGS = tf.app.flags.FLAGS

def config_initialization():
    # image shape and feature layers shape inference
    image_shape = (FLAGS.eval_image_height, FLAGS.eval_image_width)
    
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    config.init_config(image_shape, 
                       batch_size = 1, 
                       pixel_conf_threshold = FLAGS.pixel_conf_threshold,
                       link_conf_threshold = FLAGS.link_conf_threshold,
                       num_gpus = 1, 
                       model_name = FLAGS.model_name
                   )

def test():
    net, saver, image = load_net_for_inference()
    
    with tf.Session() as sess:
        saver.restore(sess, util.tf.get_latest_ckpt(FLAGS.checkpoint_path))
        files = util.io.ls(FLAGS.dataset_dir)
        
        for image_name in files:
            start = time.time()
            file_path = util.io.join_path(FLAGS.dataset_dir, image_name)
            image_data = util.img.imread(file_path)
            image_data = cv2.resize(image_data,(FLAGS.eval_image_width, FLAGS.eval_image_height), interpolation = cv2.INTER_AREA)
            link_scores, pixel_scores = predict(sess, net, image, image_data)
            link_time = time.time()
            #cProfile.runctx('post_process(image_data, pixel_scores, mask_vals)', globals(), locals())
            post_process(image_data, link_scores, pixel_scores)
            post_process_new(image_data, link_scores, pixel_scores)
            others = time.time()

            output_file = os.path.expanduser(util.get_temp_path(""))
            cv2.imwrite(output_file, image_data)
            sit_time = time.time()
            print("Score %.3f, postprocess %.3f, sit %.3f" % (link_time - start, others - link_time, sit_time - others))

def load_net_for_inference() :
    global_step = slim.get_or_create_global_step()
    with tf.name_scope('output') : # evaluation_%dx%d'%(FLAGS.eval_image_height, FLAGS.eval_image_width)):
        with tf.variable_scope(tf.get_variable_scope(), reuse = False):
            image = tf.placeholder(dtype=tf.int32, shape = [None, None, 3])
            image_shape = tf.placeholder(dtype = tf.int32, shape = [3, ])
            processed_image, _, _, _, _ = ssd_vgg_preprocessing.preprocess_image(image, 
                                                        None, None, None, None, 
                                                        out_shape = config.image_shape,
                                                        data_format = config.data_format, 
                                                        is_training = False)
            b_image = tf.expand_dims(processed_image, axis = 0)

            # build model and loss
            net = pixel_link_symbol.PixelLinkNet(b_image, is_training = False)
            #masks = pixel_link.tf_decode_score_map_to_mask_in_batch(
            #    net.pixel_pos_scores, net.link_pos_scores)
            
    sess_config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
    if FLAGS.gpu_memory_fraction < 0:
        sess_config.gpu_options.allow_growth = True
    elif FLAGS.gpu_memory_fraction > 0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction
    
    # Variables to restore: moving avg. or normal weights.
    if FLAGS.using_moving_average:
        variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore(
                tf.trainable_variables())
        variables_to_restore[global_step.op.name] = global_step
    else:
        variables_to_restore = slim.get_variables_to_restore()

    saver = tf.train.Saver(var_list = variables_to_restore)
    return net, saver, image

def predict(sess, net, image, image_data) :
    if net.link_pos_scores_add is None :
        link_scores, pixel_scores = sess.run(
                    [net.link_pos_scores, net.pixel_pos_scores],
                    feed_dict = {image: image_data})

        ps = pixel_scores
        ls = link_scores
    else :
        link_scores, pixel_scores, link_scores_add, pixel_scores_add = sess.run(
                    [net.link_pos_scores, net.pixel_pos_scores,
                    net.link_pos_scores_add, net.pixel_pos_scores_add],
                    feed_dict = {image: image_data})

        ps = (pixel_scores + pixel_scores + pixel_scores_add) / 3.0
        ls = (link_scores + link_scores + link_scores_add) / 3.0

    ls.dump("link_scores.bin")
    ps.dump("pixel_scores.bin")
    return ls, ps

def post_process_new(image_data, link_scores, pixel_scores) :
    pp = PyPixelLinkPostProcess(pixel_scores, link_scores, image_data)      
    pp.process()
    bboxes = pp.get_bounding_boxes()

    draw_bboxes(image_data, bboxes, (0,255,0))

def post_process(image_data, link_scores, pixel_scores) :
    mask_vals = pixel_link_infer.decode_batch(pixel_scores, link_scores)

    h, w, _ =image_data.shape
    def resize(img):
        return util.img.resize(img, size = (w, h), 
                                interpolation = cv2.INTER_NEAREST)
    
    def get_bboxes(mask):
        return pixel_link_infer.mask_to_bboxes(mask, image_data.shape)
    
    image_idx = 0
    pixel_score = pixel_scores[image_idx, ...]
    mask = mask_vals[image_idx, ...]

    bboxes_det = get_bboxes(mask)
    
    mask = resize(mask)
    pixel_score = resize(pixel_score)

    draw_bboxes(image_data, bboxes_det, util.img.COLOR_RGB_RED)

def draw_bboxes(img, bboxes, color):
    for bbox in bboxes:
        #pprint.pprint(bbox)
        points = np.reshape(bbox, [4, 2])
        cnts = util.img.points_to_contours(points)
        util.img.draw_contours(img, contours = cnts, 
                idx = -1, color = color, border_width = 1)

def main(_):
    config_initialization()
    test()
        
if __name__ == '__main__':
    tf.app.run()
