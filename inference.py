#encoding = utf-8
import os
import time
import datetime

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import cv2

from post_process.post_process import PyPixelLinkPostProcess

counter = 1
PIXEL_NEIGHBOUR_TYPE_4 = 'PIXEL_NEIGHBOUR_TYPE_4'
PIXEL_NEIGHBOUR_TYPE_8 = 'PIXEL_NEIGHBOUR_TYPE_8'

tf.app.flags.DEFINE_integer('eval_image_width', 1124, 'resized image width for inference')
tf.app.flags.DEFINE_integer('eval_image_height',  1536, 'resized image height for inference')
tf.app.flags.DEFINE_float('pixel_conf_threshold',  0.5, 'threshold on the pixel confidence')
tf.app.flags.DEFINE_float('link_conf_threshold',  0.5, 'threshold on the link confidence')

FLAGS = tf.app.flags.FLAGS

class configuration(object) :
    def __init__(self) :
        self.pixel_conf_threshold = FLAGS.pixel_conf_threshold,
        self.link_conf_threshold = FLAGS.link_conf_threshold,
        self.pixel_neighbour_type = PIXEL_NEIGHBOUR_TYPE_8
        self.image_shape = (FLAGS.eval_image_height , FLAGS.eval_image_width)
        self.train_image_shape = self.image_shape
        self.min_area = 300
        self.min_height = 10

        assert self.image_shape[0] % 4 == 0
        assert self.image_shape[1] % 4 == 0

config = configuration()

def interference_from_frozen() :
    image_data = cv2.imread("test/IMG_1510.jpg")
    inp = cv2.resize(image_data, (FLAGS.eval_image_width, FLAGS.eval_image_height), interpolation = cv2.INTER_CUBIC)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile('frozen/lowdim.pb', 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess :
            start = time.time()
            link_scores, pixel_scores = sess.run(
                [detection_graph.get_tensor_by_name('output/strided_slice_4:0'),
                detection_graph.get_tensor_by_name('output/strided_slice_3:0')],
                feed_dict={"output/Placeholder:0": inp.reshape(inp.shape[0], inp.shape[1], 3)} 
            )

            display_predictions(image_data, link_scores, pixel_scores, start)
            
def display_predictions(image_data, link_scores, pixel_scores, start_time) :
    net_time = time.time()

    shape = pixel_scores.shape

    pp = PyPixelLinkPostProcess(pixel_scores, link_scores)      
    pp.process()
    boxes = pp.get_bounding_boxes()
    boxes = pp.scale_boxes(image_data, boxes)

    post_time = time.time()

    draw_bboxes(image_data, boxes, (255, 0, 0))

    draw_time = time.time()

    output_file = os.path.expanduser(get_temp_path())
    cv2.imwrite(output_file, cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
    save_time = time.time()

    print "net in %.3f, postproc in %.3f, scale in %.3f, save in %.3f, total %.3f" % (
        net_time - start_time, post_time - net_time, draw_time - post_time, 
        draw_time - post_time, save_time - start_time)

def py_display_predictions(image_data, link_scores, pixel_scores) :
    net_time = time.time()
            
    mask_vals = decode_batch(pixel_scores, link_scores)
    decode_time = time.time()
    post_process(image_data, pixel_scores, mask_vals)
    post_time = time.time()
    output_file = os.path.expanduser(get_temp_path())
    cv2.imwrite(output_file, cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
    save_time = time.time()

    #print "net in %.3f, decode in %.3f, postproc in %.3f, save in %.3f, total %.3f" % (
    #    net_time - start, decode_time - net_time, post_time - decode_time, save_time - post_time, 
    #    save_time - start)

def scale_boxes(img, boxes, shape) :
    scale_y = float(img.shape[0]) / float(shape[1])
    scale_x = float(img.shape[1]) / float(shape[2])

    new_boxes = []
    for box in boxes :
        new_boxes.append([[p[0] * scale_x, p[1] * scale_y] for p in box])

    return new_boxes

def draw_bboxes(img, bboxes, color):
    for bbox in bboxes:
        points = np.reshape(bbox, [4, 2])
        cnts = points_to_contours(points)
        draw_contours(img, contours = cnts, 
                idx = -1, color = color, border_width = 1)

def post_process(image_data, pixel_scores, mask_vals) :
    h, w, _ =image_data.shape

    def get_bboxes(mask):
        return mask_to_bboxes(mask, image_data.shape)
    
    image_idx = 0
    pixel_score = pixel_scores[image_idx, ...]
    mask = mask_vals[image_idx, ...]

    bboxes_det = get_bboxes(mask)

    draw_bboxes(image_data, bboxes_det, (255, 0, 0))

def decode_batch(pixel_cls_scores, pixel_link_scores, 
                 pixel_conf_threshold = None, link_conf_threshold = None):
    start = time.time()
    if pixel_conf_threshold is None:
        pixel_conf_threshold = config.pixel_conf_threshold
    
    if link_conf_threshold is None:
        link_conf_threshold = config.link_conf_threshold

    batch_size = pixel_cls_scores.shape[0]
    batch_mask = []
    for image_idx in xrange(batch_size):
        image_pos_pixel_scores = np.squeeze(pixel_cls_scores[image_idx, :, :])
        image_pos_link_scores = np.squeeze(pixel_link_scores[image_idx, :, :])  

        shape = image_pos_pixel_scores.shape
        image_pos_link_scores = image_pos_link_scores.reshape(shape[0], shape[1], -1)

        mask = decode_image(
            image_pos_pixel_scores, image_pos_link_scores, 
            pixel_conf_threshold, link_conf_threshold
        )
        batch_mask.append(mask)

    #print "Decode batch took %.3f sec" % (time.time() - start)

    return np.asarray(batch_mask, np.int32)

def decode_image(pixel_scores, link_scores, 
                 pixel_conf_threshold, link_conf_threshold):
    mask =  decode_image_by_join(pixel_scores, link_scores, 
                 pixel_conf_threshold, link_conf_threshold)
    return mask
    
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

def is_valid_cord(x, y, w, h):
    """
    Tell whether the 2D coordinate (x, y) is valid or not.
    If valid, it should be on an h x w image
    """
    return x >=0 and x < w and y >= 0 and y < h;
        
def decode_image_by_join(pixel_scores, link_scores, 
                 pixel_conf_threshold, link_conf_threshold):
    pixel_mask = pixel_scores >= pixel_conf_threshold
    pixel_mask = pixel_scores >= pixel_conf_threshold
    link_mask = link_scores >= link_conf_threshold
    points = zip(*np.where(pixel_mask))
    h, w = np.shape(pixel_mask)
    group_mask = dict.fromkeys(points, -1)
    def find_parent(point):
        return group_mask[point]
        
    def set_parent(point, parent):
        group_mask[point] = parent
        
    def is_root(point):
        return find_parent(point) == -1
    
    def find_root(point):
        root = point
        update_parent = False
        while not is_root(root):
            root = find_parent(root)
            update_parent = True
        
        # for acceleration of find_root
        if update_parent:
            set_parent(point, root)
            
        return root
        
    def join(p1, p2):
        root1 = find_root(p1)
        root2 = find_root(p2)
        
        if root1 != root2:
            set_parent(root1, root2)
        
    def get_all():
        root_map = {}
        def get_index(root):
            if root not in root_map:
                root_map[root] = len(root_map) + 1
            return root_map[root]
        
        mask = np.zeros_like(pixel_mask, dtype = np.int32)
        for point in points:
            point_root = find_root(point)
            bbox_idx = get_index(point_root)
            mask[point] = bbox_idx
        return mask
    
    # join by link
    for point in points:
        y, x = point
        neighbours = get_neighbours(x, y)
        for n_idx, (nx, ny) in enumerate(neighbours):
            if is_valid_cord(nx, ny, w, h):
#                 reversed_neighbours = get_neighbours(nx, ny)
#                 reversed_idx = reversed_neighbours.index((x, y))
                link_value = link_mask[y, x, n_idx]# and link_mask[ny, nx, reversed_idx]
                pixel_cls = pixel_mask[ny, nx]
                if link_value and pixel_cls:
                    join(point, (ny, nx))
    
    mask = get_all()
    return mask

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

def mask_to_bboxes(mask, image_shape =  None, min_area = None, 
                   min_height = None, min_aspect_ratio = None):
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
    d = 2
    mask = cv2.resize(mask, (image_w/d, image_h/d), interpolation = cv2.INTER_NEAREST)
    
    for bbox_idx in xrange(1, max_bbox_idx + 1):
        bbox_mask = mask == bbox_idx
#         if bbox_mask.sum() < 10:
#             continue
        cnts = find_contours(bbox_mask)
        if len(cnts) == 0:
            continue
        cnt = cnts[0]
        rect, rect_area = min_area_rect(cnt)
        theta = rect[4]
        rect = [r * d for r in rect[:4]]
        rect.append(theta)
        rect_area *= (d * d)
        
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

def find_contours(mask, method = None):
    if method is None:
        method = cv2.CHAIN_APPROX_SIMPLE
    mask = np.asarray(mask, dtype = np.uint8)
    #mask = mask.copy()

    _, contours, _ = cv2.findContours(mask, mode = cv2.RETR_CCOMP, 
                               method = method)

    return contours

def points_to_contour(points):
    contours = [[list(p)]for p in points]
    return np.asarray(contours, dtype = np.int32)

def points_to_contours(points):
    return np.asarray([points_to_contour(points)])

def draw_contours(img, contours, idx = -1, color = 1, border_width = 1):
#     img = img.copy()
    cv2.drawContours(img, contours, idx, color, border_width)
    return img

def get_temp_path() :
    global counter
    dirpath = os.path.expanduser("~/temp/images")
    if not os.path.exists(dirpath) :
        os.makedirs(dirpath)
    filename = "{}-{}.png".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), counter)
    counter += 1
    return os.path.join(dirpath, filename)

def main(_):
    tf.logging.set_verbosity(tf.logging.DEBUG)
    interference_from_frozen()     
    
if __name__ == '__main__':
    tf.app.run()
