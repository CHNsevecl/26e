#include <opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "perspective.hpp"

//图形变换
void Translate(piece_data& pieces, cv::Point2f target_point,cv::Point2f leg_tight_point,double theta, int index); //拼图旋转和位移
double turn_angle(piece_data& pieces, line_data* line, size_t contour_index);
void leg_right_judgment(piece_data& pieces,const cv::Mat& warped, std::vector<std::pair<double, double>>& index, double* angle_index, int& aindex) ; //判断直角边
piece_data line_process(const cv::Mat& warped, cv::Mat& gray, cv::Mat& blurred, cv::Mat& edged, cv::Mat& binary ,cv::Mat& lineImg, std::vector<std::vector<struct line_data>>& lines); //提取拼图轮廓信息
bool draw_piece(piece_data& pieces_temp, cv::Mat& lineImg, size_t contour_index);
//用put_position之前，必须确保pieces未修改且保有旋转角度
cv::Point2f put_position(piece_data pieces_temp, rect_position& final_position, rect_length& rects_length, int max_right_contour_index, int k_index);//直角合适位置

//数据计算

//计算重心
void gravity_calculation(piece_data& pieces, cv::Mat& lineImg); 

//遍历面积，返回的拼图是包含直角的
bool Traverse_Area(piece_data& pieces, double& max_right_contour, size_t& max_right_contour_index); 

//计算面积
double polygonArea(const std::vector<cv::Point2f>& pts); 

//长度60检测 
bool l60_detect(piece_data& pieces,size_t& contour_index, size_t& line_index ); 

//长度100检测 
bool l100_detect(piece_data& pieces,size_t& contour_index, size_t& line_index ); 

//检测多边形是否重叠
bool polygonsOverlap(const std::vector<cv::Point2f>& polyA,const std::vector<cv::Point2f>& polyB,double tol = 1000.0);

//检测是否有平行的边
bool areLinesParallel(std::vector<size_t> finished_contour_index, piece_data& pieces_temp, size_t contour_index);

//越界检测
bool areOutofarea(piece_data& pieces_temp, rect_position& final_position, size_t contour_index, double tol = 5.0);
