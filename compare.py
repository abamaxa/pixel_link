#encoding = utf-8
import time
import os
import math
import pprint
import cProfile

import numpy as np
import tensorflow as tf
from datasets import dataset_factory

import util
import cv2
import pixel_link

from nets import pixel_link_symbol
from post_process.post_process import PyPixelLinkPostProcess

import pixel_link_infer

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

tf.app.flags.DEFINE_bool('using_moving_average', True, 
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
    file_path = util.io.join_path(FLAGS.dataset_dir, "img_20.jpg")
    image_data = util.img.imread(file_path)
    image_data = cv2.resize(image_data,(FLAGS.eval_image_width, FLAGS.eval_image_height), interpolation = cv2.INTER_AREA)

    link_scores = np.load("link_scores.bin")
    pixel_scores = np.load("pixel_scores.bin")

    boxes_old = post_process(image_data, link_scores, pixel_scores)
    boxes_new = post_process_new(image_data, link_scores, pixel_scores)

    output_file = os.path.expanduser(util.get_temp_path(""))
    cv2.imwrite(output_file, image_data)

    compare_boxes(boxes_old, boxes_new)

def post_process_new(image_data, link_scores, pixel_scores) :
    pp = PyPixelLinkPostProcess(pixel_scores, link_scores, image_data, False)      
    pp.process()
    bboxes = pp.get_bounding_boxes()

    draw_bboxes(image_data, bboxes, (0,255,0))
    return bboxes

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
    return bboxes_det

def draw_bboxes(img, bboxes, color):
    for bbox in bboxes:
        #pprint.pprint(bbox)
        points = np.reshape(bbox, [4, 2])
        cnts = util.img.points_to_contours(points)
        util.img.draw_contours(img, contours = cnts, 
                idx = -1, color = color, border_width = 1)

def order_points(box) :
    points = []
    i = 0
    while len(points) < 4 :
        points.append((box[i], box[i + 1]))
        i += 2

    # Sort in order of top-left, top-right, bottom-right, bottom-left
      
    points.sort(lambda a,b: -cmp(b[1], a[1]))
    points1 = points[:2]
    points1.sort(lambda a,b: -cmp(b[0], a[0]))
    points2 = points[2:]
    points2.sort(lambda a,b: -cmp(a[0], b[0]))

    points = points1 + points2
    return points

def int_boxes(boxes) :
    boxes_new = []
    for box in boxes :
        points = []
        for p in box :
            points.append((int(round(p[0])), int(round(p[1]))))
        boxes_new.append(points)

    return boxes_new

def compare_boxes(boxes_old, boxes_new) :
    
    boxes_new = int_boxes(boxes_new)

    matches = []
    for b1 in boxes_old :
        b1 = order_points(b1)
        nearest = None
        nearest_distance = 500

        for b2 in boxes_new :
            distance = 0
            for pt1, pt2 in zip(b1, b2) :
                a = pt1[0] - pt2[0]
                b = pt1[1] - pt2[1]
                distance += math.sqrt((a * a) + (b * b))

            if distance < nearest_distance :
                nearest_distance = distance
                nearest = b2

        matches.append((b1, nearest, nearest_distance))

    total_distance = 0

    for match in matches :
        print match[2]
        print match[1]
        print match[0]        
        print "-" * 80

        total_distance += match[2]

    print "Total distance", total_distance, "Num old", len(boxes_old), "Num new", len(boxes_new) 
                    

def main(_):
    config_initialization()
    test()
        
if __name__ == '__main__':
    tf.app.run()
