#include <opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#define papers_width 210
#define papers_height 297
#define gaussian_kernel_size 5

std::vector<cv::Point2f> orderPoints(std::vector<cv::Point2f> pts);
cv::Mat a4_perspective_transform(const cv::Mat& img);