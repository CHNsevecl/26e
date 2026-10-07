#include <opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#define papers_width 210
#define papers_height 297
#define gaussian_kernel_size 5
#define scale 4 // 1毫米 = 4像素

struct line_data{
    double length;
    double angle;
    double turn_angle;
    cv::Point line_vector;
    cv::Point start_point;
    cv::Point end_point;
    std::pair<double, double> near_line;
};

struct rect_length{
    double upper;
    double down;
    double left;
    double right;

};


std::vector<cv::Point2f> orderPoints(std::vector<cv::Point2f> pts);
cv::Mat a4_perspective_transform(const cv::Mat& img);
void task(const cv::Mat& warped);
void first_piece(cv::Mat& wrapped_line, line_data* line1, line_data* line2, rect_length& rect,const std::vector<std::pair<double, double>>& index);//尝试找到有长度为100mm的线段的拼图，先将其放在左上角
void second_piece(cv::Mat& wrapped_line, line_data* line1, line_data* line2, rect_length& rect, const std::vector<std::pair<double, double>>& index);//第二块拼图，将其放在左下角