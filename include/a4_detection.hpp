#pragma once

#include <opencv4/opencv2/opencv.hpp>
#include <vector>

struct A4Detection {
    bool found = false;
    std::vector<cv::Point2f> corners; // 顺时针，短边对应 A4 的宽。
    double score = 0.0;
    cv::Mat dark_mask;
    cv::Mat edges;
    cv::Mat debug_image;
};

// 无摄像头、串口或文件副作用，可用于现场图片的离线检测。
A4Detection detect_black_a4(const cv::Mat& image);
