import tensorflow as tf
import numpy as np
import cv2

from pixel_link import *

import config
import util

#============================Decode Begin=============================

def tf_decode_score_map_to_mask_in_batch(pixel_cls_scores, pixel_link_scores):
    masks = tf.py_func(decode_batch, 
                       [pixel_cls_scores, pixel_link_scores], tf.int32)
    b, h, w = pixel_cls_scores.shape.as_list()
    masks.set_shape([b, h, w])
    return masks 

def decode_batch(pixel_cls_scores, pixel_link_scores, 
                 pixel_conf_threshold = None, link_conf_threshold = None):
    if pixel_conf_threshold is None:
        pixel_conf_threshold = config.pixel_conf_threshold
    
    if link_conf_threshold is None:
        link_conf_threshold = config.link_conf_threshold

    batch_size = pixel_cls_scores.shape[0]
    batch_mask = []
    for image_idx in xrange(batch_size):
        image_pos_pixel_scores = pixel_cls_scores[image_idx, :, :]
        image_pos_link_scores = pixel_link_scores[image_idx, :, :]    
        
        shape = image_pos_pixel_scores.shape
        image_pos_link_scores = image_pos_link_scores.reshape(shape[0], shape[1], -1)

        mask = decode_image(
            image_pos_pixel_scores, image_pos_link_scores, 
            pixel_conf_threshold, link_conf_threshold
        )
        batch_mask.append(mask)

    return np.asarray(batch_mask, np.int32)

# @util.dec.print_calling_in_short
# @util.dec.timeit
def decode_image(pixel_scores, link_scores, 
                 pixel_conf_threshold, link_conf_threshold):
    if config.decode_method == DECODE_METHOD_join:
        mask =  decode_image_by_join(pixel_scores, link_scores, 
                 pixel_conf_threshold, link_conf_threshold)
        return mask
    elif config.decode_method == DECODE_METHOD_border_split:
        return decode_image_by_border(pixel_scores, link_scores, 
                 pixel_conf_threshold, link_conf_threshold)
    else:
        raise ValueError('Unknow decode method:%s'%(config.decode_method))


import pyximport; pyximport.install()    
from pixel_link_decode import decode_image_by_join

def min_area_rect(cnt):
    """
    Args:
        xs: numpy ndarray with shape=(N,4). N is the number of oriented bboxes. 4 contains [x1, x2, x3, x4]
        ys: numpy ndarray with shape=(N,4), [y1, y2, y3, y4]
            Note that [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] can represent an oriented bbox.
    Return:
        the oriented rects sorrounding the box, in the format:[cx, cy, w, h, theta]. 
    """
    rect = cv2.minAreaRect(cnt)
    cx, cy = rect[0]
    w, h = rect[1]
    theta = rect[2]
    box = [cx, cy, w, h, theta]
    return box, w * h

def rect_to_xys(rect, image_shape):
    """Convert rect to xys, i.e., eight points
    The `image_shape` is used to to make sure all points return are valid, i.e., within image area
    """
    h, w = image_shape[0:2]
    def get_valid_x(x):
        if x < 0:
            return 0
        if x >= w:
            return w - 1
        return x
    
    def get_valid_y(y):
        if y < 0:
            return 0
        if y >= h:
            return h - 1
        return y
    
    rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
    points = cv2.boxPoints(rect)
    points = np.int0(points)
    for i_xy, (x, y) in enumerate(points):
        x = get_valid_x(x)
        y = get_valid_y(y)
        points[i_xy, :] = [x, y]
    points = np.reshape(points, -1)
    return points

# @util.dec.print_calling_in_short
# @util.dec.timeit
def mask_to_bboxes(mask, image_shape =  None, min_area = None, 
                   min_height = None, min_aspect_ratio = None):
    import config
    feed_shape = config.train_image_shape
    
    if image_shape is None:
        image_shape = feed_shape
        
    image_h, image_w = image_shape[0:2]
    
    if min_area is None:
        min_area = config.min_area
        
    if min_height is None:
        min_height = config.min_height
    bboxes = []
    max_bbox_idx = mask.max()
    d = 1
    mask = util.img.resize(img = mask, size = (image_w/d, image_h/d), 
                           interpolation = cv2.INTER_NEAREST)
    
    for bbox_idx in xrange(1, max_bbox_idx + 1):
        bbox_mask = mask == bbox_idx
#         if bbox_mask.sum() < 10:
#             continue
        cnts = util.img.find_contours(bbox_mask)
        if len(cnts) == 0:
            continue
        cnt = cnts[0]
        rect, rect_area = min_area_rect(cnt)
        theta = rect[4]
        rect = [r * float(d) for r in rect[:4]]
        rect.append(theta)
        rect_area *= float(d * d)
        
        w, h = rect[2:-1]
        if min(w, h) < min_height:
            continue
        
        if rect_area < min_area:
            continue
        
#         if max(w, h) * 1.0 / min(w, h) < 2:
#             continue
        xys = rect_to_xys(rect, image_shape)
        bboxes.append(xys)
        
    return bboxes


#============================Decode End===============================
