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
    cv::Point2f line_vector;
    cv::Point2f start_point;
    cv::Point2f end_point;
    std::pair<double, double> near_line;
    bool leg_right_angle;
};

struct piece_data{
    std::vector<std::vector<cv::Point2f>> pieces_point;
    std::vector<cv::Point2f> center_gravity;
    std::vector<std::vector<cv::Point2f>> gra_p_vector; //重心到点的向量
    std::vector<std::vector<struct line_data>> pieces_lines;
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
void Translate(piece_data& pieces, cv::Point2f target_point,cv::Point2f leg_tight_point,double theta, int index); //拼图旋转和位移
void turn_angle(line_data* line,int index);