# python setup.py build_ext --inplace

import numpy as np
cimport numpy as np # for np.ndarray
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "opencv2/core/core.hpp" namespace "cv":
  cdef cppclass Point2f:
    Point2f() except +
    Point2f(float, float) except +
    float x
    float y

cdef extern from "PixelLinkPostProcess.cpp":
    pass
        
cdef extern from "PixelLinkPostProcess.h" :
    cdef cppclass PixelLinkPostProcess:
        PixelLinkPostProcess(float* pixel_scores,
                         float* link_scores,
                         int rows, int cols,
                         int image_width, int image_height,
                         float pixel_conf_threshold,
                         float link_conf_threshold,
                         bool fast_mode,
                         int min_dimension,
                         int min_area) except +
        void process()
        void calculate_boxes(vector[vector[Point2f]]& boxes) const
        void calculate_hulls(vector[vector[Point2f]]& hulls) const

cdef class PyPixelLinkPostProcess:
    cdef PixelLinkPostProcess* c_post_processor
    cdef int __rows
    cdef int __cols
    
    def __cinit__(self, pixel_scores, link_scores, image, fast_mode = False,
                pixel_conf_threshold = 0.5, link_conf_threshold = 0.5,
                min_dimension = 10, min_area = 300):
        shape = pixel_scores.shape
        self.__rows = shape[1]
        self.__cols = shape[2]

        if not pixel_scores.flags['C_CONTIGUOUS']:
            pixel_scores = np.ascontiguousarray(pixel_scores)

        if not link_scores.flags['C_CONTIGUOUS']:
            link_scores = np.ascontiguousarray(link_scores)

        cdef float [:,:,::1] c_pixel_scores = pixel_scores
        cdef float [:,:,::1] c_link_scores = link_scores
        
        self.c_post_processor = new PixelLinkPostProcess(
            &c_pixel_scores[0][0][0], &c_link_scores[0][0][0],
            self.__rows, self.__cols, image.shape[1], image.shape[0],
            pixel_conf_threshold, link_conf_threshold,
            fast_mode, min_dimension, min_area)
 
    def __dealloc__(self):
        del self.c_post_processor
        
    def process(self):
        self.c_post_processor.process()
        
    def get_bounding_boxes(self) :
        cdef vector[vector[Point2f]] points
        cdef vector[Point2f] box
        cdef Point2f box_point

        self.c_post_processor.calculate_boxes(points)

        py_boxes = []
        for i in range(points.size()) :
            box = points.at(i)
            py_box = []
            for n in range(box.size()) :
                box_point = box.at(n)
                py_box.append((box_point.x, box_point.y))

            py_boxes.append(py_box)
            
        return py_boxes

    # def scale_boxes(self, img, boxes) :
    #     scale_y = float(img.shape[0]) / float(self.__rows)
    #     scale_x = float(img.shape[1]) / float(self.__cols)

    #     new_boxes = []
    #     for box in boxes :
    #         new_boxes.append(self.scale_box(box, scale_x, scale_y))

    #     return new_boxes

    # def scale_box(self, box, scale_x, scale_y) :
    #     # Sort in order of top-left, top-right, bottom-right, bottom-left
    #     new_box = [None] * 4 
    #     new_box[0] = [box[0][0] * scale_x,  box[0][1] * scale_y]
    #     new_box[1] = [(box[1][0] * scale_x) + int(scale_x) - 1,  box[1][1] * scale_y]
    #     new_box[2] = [(box[2][0] * scale_x) + int(scale_x) - 1,  (box[2][1] * scale_y) + int(scale_y) - 1]
    #     new_box[3] = [box[3][0] * scale_x,  (box[3][1] * scale_y) + int(scale_y) - 1]
    #     return new_box
