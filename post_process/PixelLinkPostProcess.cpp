//
//  PixelLinkPosProcess.cpp
//  CoreMLSSD
//

#include "PixelLinkPostProcess.h"

void inline get_neighbours_8(int x, int y, PixelPoint* result);
void inline get_neighbours_4(int x, int y, PixelPoint* result);
inline bool is_valid_cord(int x, int y, int w, int h);
void set_corners(std::vector<cv::Point2f>& corners);

PixelMap::PixelMap(int _rows, int _cols) :
 cols(_cols),
 xscale(1.0),
 yscale(1.0)
{
    map = cv::Mat::zeros(_rows, _cols, CV_32SC1);
}

PixelMap::PixelMap(cv::Mat& _map, int _cols, float _xscale, float _yscale) :
    cols(_cols),
    xscale(_xscale),
    yscale(_yscale),
    map(_map)
{
}

void PixelMap::join(const PixelPoint& pt1, const PixelPoint& pt2) {
    auto root1 = find_root(pt1);
    auto root2 = find_root(pt2);
    
    if (root1 != root2)
        set_parent(root1, root2);
}

int PixelMap::point_to_ordinal(const PixelPoint& pt) const {
    if (xscale == 1.0 && yscale == 1.0)
        return (pt.y * cols) + pt.x;
    else
        return ((int)round((float)pt.y / yscale) * cols) + (int)round(pt.x / xscale);
}

PixelPoint PixelMap::value_to_point(int val) const {
    int ordinal = val - 1;
    assert(ordinal >= 0);
    int y = ordinal / cols;
    int x = ordinal - (y * cols);
    if (xscale == 1.0 && yscale == 1.0)
        return PixelPoint(x, y);
    else
        return PixelPoint((int)round(x * xscale), (int)round(y * yscale));
}

int PixelMap::find(const PixelPoint& key) const {
    if (key == ROOT_NODE)
        return 0;
    
    return map.at<int>(key.y, key.x);
}

int PixelMap::at(int y, int x) const {
    return map.at<int>(y, x);
}

PixelPoint PixelMap::find_parent(const PixelPoint& pt) const {
    auto itr = find(pt);
    if (!itr)
        return ROOT_NODE;
    else
        return value_to_point(itr);
}

PixelPoint PixelMap::find_root(const PixelPoint& pt) {
    PixelPoint root = pt;
    while (!is_root(root)) {
        PixelPoint parent = find_parent(root);
        if (parent == root)
            break;
        root = parent;
    }
    
    if (pt != root)
        set_parent(pt, root);
    
    return root;
}

bool PixelMap::is_root(const PixelPoint& pt) const {
    return find_parent(pt) == ROOT_NODE;
}

void PixelMap::set_parent(const PixelPoint& child, const PixelPoint& parent) {
    int value = point_to_ordinal(parent) + 1;
    int offset = point_to_ordinal(child);
    int* ptr = (int*)map.ptr();
    ptr += offset;
    if (*ptr != value)
        *ptr = value;
}

PixelMap PixelMap::resize(int new_width, int new_height) {
    float xscale = (float)new_width / (float)map.cols;
    float yscale = (float)new_height / (float)map.rows;
    cv::Mat resized_map;
    cv::resize(map, resized_map, cv::Size(new_width, new_height), 0, 0, cv::INTER_NEAREST);
    return PixelMap(resized_map, cols, xscale, yscale);
}

PixelLinkPostProcess::PixelLinkPostProcess
(
    void* _pixel_scores,
    void* _link_scores,
    int _rows,
    int _cols,
    int _image_width,
    int _image_height,
    float _pixel_conf_threshold,
    float _link_conf_threshold,
    bool _fast_mode,
    int _min_dimension,
    int _min_area
) :
    pixel_scores(cv::Mat(_rows, _cols, PP_MAT_DATA, _pixel_scores)),
    link_scores(cv::Mat(_rows * _cols, 8, PP_MAT_DATA, _link_scores)),
    pixel_map(_rows, _cols),
    rows(_rows),
    cols(_cols),
    image_width(_image_width),
    image_height(_image_height),
    pixel_conf_threshold(_pixel_conf_threshold),
    link_conf_threshold(_link_conf_threshold),
    fast_mode(_fast_mode),
    min_dimension(_min_dimension),
    min_area(_min_area),
    xscale((float)_image_width / (float)_cols),
    yscale((float)_image_height / (float)_rows)
{
}

void PixelLinkPostProcess::process() {
    decode_image_by_join();
    if (fast_mode)
        gather_regions_fast();
    else
        gather_regions();
}

void PixelLinkPostProcess::decode_image_by_join() {
    const int num_links = 8;
    
    for (int y = 0;y < pixel_scores.rows;y++) {
        for (int x = 0;x < pixel_scores.cols;x++) {
            float pixel_score = pixel_scores.at<ScoreDataType>(y, x);
            if (pixel_score < pixel_conf_threshold)
                continue;
        
            PixelPoint points[num_links];
            get_neighbours(x, y, points);
            
            for (int l = 0;l < num_links;l++) {
                const PixelPoint& pt = points[l];
                if (!is_valid_cord(pt.x, pt.y, pixel_scores.cols, pixel_scores.rows))
                    continue;

                if (pixel_scores.at<ScoreDataType>(pt.y, pt.x) < pixel_conf_threshold)
                    continue;
                
                float link_value = link_scores.at<ScoreDataType>((y * cols) + x, l);
                if (link_value < link_conf_threshold)
                    continue;
                
                // Was y,x order
                pixel_map.join(PixelPoint(x,y), pt);
            }
        }
    }
}

void PixelLinkPostProcess::gather_regions() {
    for (int y = 0;y < pixel_scores.rows;y++) {
        for (int x = 0;x < pixel_scores.cols;x++) {
            int value = pixel_map.at(y, x);
            if (!value)
                continue;
            
            pixel_map.find_root(PixelPoint(x, y));
        }
    }

    //cv::imwrite("test.png", pixel_map.map);

    PixelMap resized_scores = pixel_map.resize(image_width, image_height);
    gathered_regions.clear();
    
    for (int y = 0;y < image_height;y++) {
        for (int x = 0;x < image_width;x++) {
            int value = resized_scores.at(y, x);
            if (!value)
                continue;
            
            PixelPoint point = PixelPoint(x, y);
            PixelPoint root = resized_scores.find_root(resized_scores.value_to_point(value));
            
            auto itr = gathered_regions.find(root);
            if (itr != gathered_regions.end()) {
                (*itr).second.push_back(point);
            }
            else {
                std::vector<PixelPoint> points = {root,point};
                gathered_regions.emplace(root, points);
            }
        }
    }
}

void PixelLinkPostProcess::gather_regions_fast() {
    gathered_regions.clear();
    for (int y = 0;y < pixel_scores.rows;y++) {
        for (int x = 0;x < pixel_scores.cols;x++) {
            int value = pixel_map.at(y, x);
            if (!value)
                continue;
            
            PixelPoint point = PixelPoint(x, y);
            PixelPoint root = pixel_map.find_root(pixel_map.value_to_point(value));
            
            auto itr = gathered_regions.find(root);
            if (itr != gathered_regions.end()) {
                (*itr).second.push_back(point);
            }
            else {
                std::vector<PixelPoint> points = {root,point};
                gathered_regions.emplace(root, points);
            }
        }
    }
    
    for (auto& pair : gathered_regions) {
        scale_points(pair.second);
    }
}

void PixelLinkPostProcess::calculate_hulls(PixelLinkList& boxes) const {
    const int    MIN_PIXEL = 4;
    boxes.clear();
    
    for (auto& pair : gathered_regions) {
        auto& points = pair.second;
        if (points.size() < MIN_PIXEL)
            continue;

        std::vector<int> hull;
        cv::convexHull(points, hull, true);

        std::vector<cv::Point2f> hull_points;
        for (auto index : hull) {
            hull_points.push_back(cv::Point2f(points[index]));
        }
        
        boxes.push_back(hull_points);
    }
}

void PixelLinkPostProcess::calculate_boxes(PixelLinkList& boxes) const {
    const int    MIN_PIXEL = 4;
    
    boxes.clear();
    
    for (auto& pair : gathered_regions) {
        auto& points = pair.second;
        if (points.size() < MIN_PIXEL)
            continue;

        cv::RotatedRect rect = cv::minAreaRect(points);
        
        if (std::min(rect.size.width, rect.size.height) < min_dimension)
            continue;
         
        if (rect.size.width * rect.size.height < min_area)
            continue;
        
        std::vector<cv::Point2f> box_points;
        box_points.resize(4);
        rect.points(box_points.data());
        set_corners(box_points);
        boxes.push_back(pad_boxes(box_points));
    }
}

PixelList& PixelLinkPostProcess::scale_points(PixelList& points) const {
    for (auto& point : points) {
        /*point.x = (int)(xscale / 2) + (int)round(point.x * xscale);
        point.y = (int)(yscale / 2) + (int)round(point.y * yscale);*/
        point.x = (int)round(point.x * xscale);
        point.y = (int)round(point.y * yscale);
    }
    return points;
}

Point2fList& PixelLinkPostProcess::pad_boxes(Point2fList& points) const {
    if (fast_mode) {
        points[1].x += int(xscale) - 1;
        points[2].x += int(xscale) - 1;
        points[2].y += int(yscale) - 1;
        points[3].y += int(yscale) - 1;
    }
    return points;
}

void PixelLinkPostProcess::get_neighbours(int x, int y, PixelPoint* result) const
{
    if (pixel_neighbour_type == PIXEL_NEIGHBOUR_TYPE_4) {
        get_neighbours_4(x, y, result);
    }
    else {
        get_neighbours_8(x, y, result);
    }
}

void inline get_neighbours_8(int x, int y, PixelPoint* result) {
    /*
     Get 8 neighbours of point(x, y)
     */
    PixelPoint neighbors[] = {
        {x - 1, y - 1}, {x, y - 1}, {x + 1, y - 1},
        {x - 1, y},                 {x + 1, y},
        {x - 1, y + 1}, {x, y + 1}, {x + 1, y + 1}};
    memcpy(result, neighbors, sizeof(neighbors));
}

void inline get_neighbours_4(int x, int y, PixelPoint* result) {
    PixelPoint neighbors[] = {{x - 1, y}, {x + 1, y}, {x, y + 1}, {x, y - 1}};
    memcpy(result, neighbors, sizeof(neighbors));
}

inline bool is_valid_cord(int x, int y, int w, int h) {
    /*
     Tell whether the 2D coordinate (x, y) is valid or not.
     If valid, it should be on an h x w image
     */
    return x >=0 && x < w && y >= 0 && y < h;
}

void set_corners(std::vector<cv::Point2f>& corners) {
    // Sort in order of top-left, top-right, bottom-right, bottom-left
      
    std::sort(corners.begin(),corners.end(),[](cv::Point2f& a,cv::Point2f& b) {
        return a.y < b.y;
    });
    
    // top most
    std::sort(corners.begin(),corners.begin()+2,[](cv::Point2f& a,cv::Point2f& b) {
        return a.x < b.x;
    });
    
    // bottom most
    std::sort(corners.begin() + 2, corners.end(),[](cv::Point2f& a,cv::Point2f& b) {
        return a.x > b.x;
    });
}
