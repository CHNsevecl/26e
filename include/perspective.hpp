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
    cv::Point2f start_point;
    cv::Point2f end_point;
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
cv::Mat a4_perspective_transform(const cv::Mat& img);
void task(const cv::Mat& warped);
void gravity_calculation(piece_data& pieces, cv::Mat& lineImg); //计算重心
double polygonArea(const std::vector<cv::Point2f>& pts); //计算面积
void Translate(piece_data& pieces, cv::Point2f target_point,cv::Point2f leg_tight_point,double theta, int index); //拼图旋转和位移
void leg_right_judgment(piece_data& pieces,const cv::Mat& warped, std::vector<std::pair<double, double>>& index, double* angle_index, int& aindex) ; //判断直角边
piece_data line_process(const cv::Mat& warped, cv::Mat& gray, cv::Mat& blurred, cv::Mat& edged, cv::Mat& binary ,cv::Mat& lineImg, std::vector<std::vector<struct line_data>>& lines); //提取拼图轮廓信息
double turn_angle(line_data* line,int index);
cv::Point2f put_position(piece_data pieces_temp, rect_position& final_position, rect_length& rects_length, int max_right_contour_index, int k_index);//直角合适位置