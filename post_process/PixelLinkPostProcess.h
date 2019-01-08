//
//  PixelLinkPosProcess.hpp
//  CoreMLSSD
//
//  Created by Chris Morgan on 26/7/18.
//  Copyright © 2018 Chris Morgan. All rights reserved.
//

#ifndef PixelLinkPosProcess_hpp
#define PixelLinkPosProcess_hpp

#if __APPLE__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdocumentation"
#endif

#include <opencv2/opencv.hpp>

#if __APPLE__
#pragma clang diagnostic pop
#endif

#include <cmath>
#include <algorithm>
#include <unordered_map>

enum NEIGHBOUR_TYPE {
    PIXEL_NEIGHBOUR_TYPE_4,
    PIXEL_NEIGHBOUR_TYPE_8
};

typedef cv::Point2i PixelPoint;
typedef std::vector<PixelPoint> PixelList;
typedef std::vector<cv::Point2f> Point2fList;
typedef std::vector<Point2fList> PixelLinkList;

struct PointHash {
    std::size_t operator()(const PixelPoint& k) const noexcept {
        const int MAX_PIXEL_POINT_DIM = 50000;
        return k.x + k.y * MAX_PIXEL_POINT_DIM;
    }
};

typedef std::unordered_map<PixelPoint, PixelList, PointHash> PixelRegions;
typedef std::unordered_map<PixelPoint, PixelPoint, PointHash> Point2Point;

class PixelMap {
public :
    PixelMap(int rows, int cols);
    PixelMap(cv::Mat& map, int _cols, float xscale, float yscale);
    void join(const PixelPoint& pt1, const PixelPoint& pt2);
    PixelPoint find_root(const PixelPoint& pt);
    int at(int y, int x) const;
    PixelPoint value_to_point(int ordinal) const;
    
    PixelMap resize(int new_width, int new_height);
    
    cv::Mat map;
    
private:
    PixelPoint find_parent(const PixelPoint& pt) const;
    bool is_root(const PixelPoint& pt) const;
    void set_parent(const PixelPoint& child, const PixelPoint& parent);
    int point_to_ordinal(const PixelPoint& pt) const;
    
    int find(const PixelPoint& key) const;
    
    int cols;
    float xscale;
    float yscale;
    
    const PixelPoint ROOT_NODE = PixelPoint(-1,-1);
};

#ifdef FLOAT32_DATA
typedef float ScoreDataType;
#define PP_MAT_DATA CV_32FC1
#else
typedef double ScoreDataType;
#define PP_MAT_DATA CV_64FC1
#endif

class PixelLinkPostProcess {
public:
    PixelLinkPostProcess(void* pixel_scores,
                         void* link_scores,
                         int rows, int cols,
                         int image_width,
                         int image_height,
                         float pixel_conf_threshold,
                         float link_conf_threshold,
                         bool fast_mode = false,
                         int min_dimension = 10,
                         int min_area = 300);
    
    void process();
    void calculate_boxes(PixelLinkList& boxes) const;
    void calculate_hulls(PixelLinkList& hulls) const;
    
private:
    void decode_batch();    
    void get_neighbours(int x, int y, PixelPoint* result) const;
    void decode_image_by_join();
    void gather_regions();
    void gather_regions_fast();
    PixelList& scale_points(PixelList& points) const;
    Point2fList& pad_boxes(Point2fList& points) const;

    const cv::Mat pixel_scores;
    const cv::Mat link_scores;
    
    PixelMap pixel_map; 

    const int rows;
    const int cols;
    
    const int image_width;
    const int image_height;
          
    PixelRegions gathered_regions;
    
    const float pixel_conf_threshold;
    const float link_conf_threshold;

    const bool fast_mode;
    const int min_dimension;
    const int min_area;

    const float xscale;
    const float yscale;

    const NEIGHBOUR_TYPE pixel_neighbour_type = PIXEL_NEIGHBOUR_TYPE_8;
};

#endif /* PixelLinkPosProcess_hpp */
