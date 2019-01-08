import time

import tensorflow as tf
import numpy as np
import cv2
import util

PIXEL_CLS_WEIGHT_all_ones = 'PIXEL_CLS_WEIGHT_all_ones' 
PIXEL_CLS_WEIGHT_bbox_balanced = 'PIXEL_CLS_WEIGHT_bbox_balanced'
PIXEL_NEIGHBOUR_TYPE_4 = 'PIXEL_NEIGHBOUR_TYPE_4'
PIXEL_NEIGHBOUR_TYPE_8 = 'PIXEL_NEIGHBOUR_TYPE_8'
DECODE_METHOD_join = 'DECODE_METHOD_join'

import config

def get_neighbours_8(x, y):
    """
    Get 8 neighbours of point(x, y)
    """
    return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), \
        (x - 1, y),                 (x + 1, y),  \
        (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]


def get_neighbours_4(x, y):
    return [(x - 1, y), (x + 1, y), (x, y + 1), (x, y - 1)]


def get_neighbours(x, y):
    neighbour_type = config.pixel_neighbour_type
    if neighbour_type == PIXEL_NEIGHBOUR_TYPE_4:
        return get_neighbours_4(x, y)
    else:
        return get_neighbours_8(x, y)
    
def get_neighbours_fn():
    neighbour_type = config.pixel_neighbour_type
    if neighbour_type == PIXEL_NEIGHBOUR_TYPE_4:
        return get_neighbours_4, 4
    else:
        return get_neighbours_8, 8

def is_valid_cord(x, y, w, h):
    """
    Tell whether the 2D coordinate (x, y) is valid or not.
    If valid, it should be on an h x w image
    """
    return x >=0 and x < w and y >= 0 and y < h;



