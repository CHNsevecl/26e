#pragma once

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
    cv::Point2f line_vector;
    size_t start_point;
    size_t end_point;
    std::pair<double, double> near_line;
    bool line_right_angle;
    size_t last_line_index;
};

struct piece_data{
    std::vector<std::vector<cv::Point2f>> pieces_point;
    std::vector<cv::Point2f> center_gravity;
    std::vector<std::vector<cv::Point2f>> gra_p_vector; //重心到点的向量
    std::vector<std::vector<struct line_data>> pieces_lines;
    std::vector<double> turn_angle;
    std::vector<std::vector<bool>> pieces_lines_right_point;
};

struct rect_length{
    std::pair<cv::Point2f, cv::Point2f> up;
    std::pair<cv::Point2f, cv::Point2f> down;
    std::pair<cv::Point2f, cv::Point2f> left;
    std::pair<cv::Point2f, cv::Point2f> right;
    double up_length;
    double down_length;
    double left_length;
    double right_length;
};

struct rect_position{
    cv::Point2f upper_left;
    cv::Point2f upper_right;
    cv::Point2f down_left;
    cv::Point2f down_right;
};

std::vector<cv::Point2f> orderPoints(std::vector<cv::Point2f> pts);

void task(const cv::Mat& warped);
cv::Mat a4_perspective_transform(const cv::Mat& img);

