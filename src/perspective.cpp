#include "perspective.hpp"

using namespace cv;
using namespace std;

// 对四个角点排序：左上、右上、右下、左下
vector<Point2f> orderPoints(vector<Point2f> pts) {
    vector<Point2f> ordered(4);
    // 按 x + y 之和排序，最小的为左上，最大的为右下
    vector<float> sum;
    for (auto& p : pts) sum.push_back(p.x + p.y);
    ordered[0] = pts[min_element(sum.begin(), sum.end()) - sum.begin()]; // 左上
    ordered[2] = pts[max_element(sum.begin(), sum.end()) - sum.begin()]; // 右下

    // 按 x - y 之差排序，最小的为左下，最大的为右上
    vector<float> diff;
    for (auto& p : pts) diff.push_back(p.x - p.y);
    ordered[1] = pts[max_element(diff.begin(), diff.end()) - diff.begin()]; // 右上
    ordered[3] = pts[min_element(diff.begin(), diff.end()) - diff.begin()]; // 左下

    return ordered;
}

Mat a4_perspective_transform(const Mat& img) {
    Mat gray, blurred, edged;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(gaussian_kernel_size, gaussian_kernel_size), 0); //高斯模糊
    Canny(blurred, edged, 50, 150);

    // 2. 找轮廓
    vector<vector<Point>> contours;
    findContours(edged, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 3. 筛选面积最大的四边形
    vector<Point2f> srcPts;
    double maxArea = 0;

    for (auto& c : contours) {
        double area = contourArea(c);
        if (area < 10000) continue; // 过滤小轮廓

        // 多边形逼近
        vector<Point> approx;
        double peri = arcLength(c, true);
        approxPolyDP(c, approx, 0.02 * peri, true);

        if (approx.size() == 4 && area > maxArea) {
            maxArea = area;
            srcPts.clear();
            for (auto& p : approx) srcPts.push_back(Point2f(p.x, p.y));
        }
    }

    if (srcPts.size() != 4) {
        cout << "未检测到四边形，请尝试手动选点或调整参数" << endl;
        return Mat();
    }

    // 4. 排序角点
    srcPts = orderPoints(srcPts);

    // 5. A4 纸真实坐标（单位：mm），按 左上→右上→右下→左下
    
    vector<Point2f> dstPts = {
        Point2f(0, 0),
        Point2f(papers_width*scale, 0),
        Point2f(papers_width*scale, papers_height*scale),
        Point2f(0, papers_height*scale)
    };

    // 6. 计算透视变换矩阵
    Mat M = getPerspectiveTransform(srcPts, dstPts);

    // 7. 执行透视变换，输出尺寸 210×297（1像素 = 1毫米）
    
    Mat warped;
    warpPerspective(img, warped, M, Size(papers_width * scale, papers_height * scale));

    // 8. 削去最外圈 5 像素
    int margin = 5;
    cv::Rect roi(margin, margin,
                 warped.cols - 2 * margin,
                 warped.rows - 2 * margin);
    warped = warped(roi).clone();  // clone() 保证返回的是独立内存，不依赖原图

    // 9. 保存结果
    imwrite("a4_warped.jpg", warped);
    return warped;
}

void task(const cv::Mat& warped){
    cv::Mat gray, blurred, edged, binary;
    cv::cvtColor(warped, gray, cv::COLOR_BGR2GRAY);
    

    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 5);  // 核可适当加大
    cv::imwrite("blurred.png", blurred);
    cv::threshold(blurred, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::imwrite("origin_binary.png", binary);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1));
    cv::dilate(binary, binary, kernel);
    cv::Canny(binary, edged, 0, 10);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edged, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat lineImg = cv::Mat::zeros(cv::Size(papers_width*scale, papers_height*scale), CV_8UC1);

    std::vector<std::vector<struct line_data>> lines(contours.size());
    struct piece_data pieces;
    pieces.pieces_point.resize(contours.size());
    pieces.center_gravity.resize(contours.size());
    pieces.gra_p_vector.resize(pieces.pieces_point.size());

    std::cout << "Contours found: " << contours.size() << std::endl;
    for (size_t i = 0; i < contours.size(); i++) {
        std::vector<cv::Point> approx;
        double peri = cv::arcLength(contours[i], true);
        cv::approxPolyDP(contours[i], approx, 0.01 * peri, true);
        
        lines[i].reserve(approx.size());
        for (size_t j = 0; j < approx.size(); j++) {
            cv::Point2f p1 = approx[j];
            cv::Point2f p2 = approx[(j + 1) % approx.size()];
            double length = cv::norm(p2 - p1);
            double angle = std::atan2(p2.y - p1.y, p2.x - p1.x) * 180 / CV_PI;
            line_data ld{length, 0.0, 0.0, p2 - p1, p2 ,p1, std::make_pair(-1, -1),false};
            lines[i].push_back(ld);
            pieces.pieces_point[i].push_back(p1);
            cv::line(lineImg, p1, p2, cv::Scalar(255), 1);

            cv::imwrite("Line Segment.png", lineImg);
        }
    }
    

    //计算重心
    std::vector<std::vector<double>> ci(pieces.pieces_point.size());
    std::vector<std::vector<double>> sx(pieces.pieces_point.size());
    std::vector<std::vector<double>> sy(pieces.pieces_point.size());
    for(size_t i = 0; i < pieces.pieces_point.size(); i++) {
        ci[i].resize(pieces.pieces_point[i].size());
        sx[i].resize(pieces.pieces_point[i].size());
        sy[i].resize(pieces.pieces_point[i].size());
        for(size_t j = 0; j < pieces.pieces_point[i].size(); j++) {
            ci[i][j] = pieces.pieces_point[i][j].x * pieces.pieces_point[i][(j+1)%pieces.pieces_point[i].size()].y - pieces.pieces_point[i][(j+1)%pieces.pieces_point[i].size()].x * pieces.pieces_point[i][j].y;
        }
    }
    
    for(size_t i = 0; i < pieces.pieces_point.size(); i++) {
        for (size_t j = 0; j < pieces.pieces_point[i].size(); j++) {
            
                sx[i][j] = (pieces.pieces_point[i][j].x + pieces.pieces_point[i][(j+1)%pieces.pieces_point[i].size()].x) * ci[i][j];
                sy[i][j] = (pieces.pieces_point[i][j].y + pieces.pieces_point[i][(j+1)%pieces.pieces_point[i].size()].y) * ci[i][j];
            
        }
    }

    for(size_t i = 0; i < pieces.pieces_point.size(); i++) {
        double sum_ci = 0;
        double sum_sx = 0;
        double sum_sy = 0;
        for(size_t j = 0; j < pieces.pieces_point[i].size(); j++) {
            sum_ci += ci[i][j];
            sum_sx += sx[i][j];
            sum_sy += sy[i][j];
        }
        pieces.center_gravity[i].x = sum_sx / (3 * sum_ci);
        pieces.center_gravity[i].y = sum_sy / (3 * sum_ci);
        cv::circle(lineImg, pieces.center_gravity[i], 3, cv::Scalar(255), -1);
    }

    //计算重心到点的向量
    for(size_t i = 0; i < pieces.pieces_point.size(); i++) {
        for(size_t j = 0; j < pieces.pieces_point[i].size(); j++) {
            cv::Point2f vector = pieces.pieces_point[i][j] - pieces.center_gravity[i];
            pieces.gra_p_vector[i].push_back(vector);
            cv::circle(lineImg, pieces.pieces_point[i][j], 2, cv::Scalar(255), -1);
            cv::line(lineImg, pieces.center_gravity[i], pieces.center_gravity[i] + vector, cv::Scalar(255), 1);
            cv::imwrite("lines.png", lineImg);
        }   
    }
    pieces.pieces_lines = lines;

    //与同轮廓上一个索引的线的夹角，判断是否为直角边
    std::vector<std::pair<double, double>> index;
    double angle_index[100];
    int aindex = 0;
    for(size_t i = 0; i < pieces.pieces_lines.size(); i++) {
        for(size_t j = 0; j < pieces.pieces_lines[i].size(); j++) {
            line_data* line2 = nullptr;
            cv::Point p1 = pieces.pieces_lines[i][j].line_vector;
            cv::Point p2;
            if(j != 0) {
                line2 = &pieces.pieces_lines[i][j-1];
                p2 = line2->line_vector;
            }
            else {
                line2 = &pieces.pieces_lines[i][pieces.pieces_lines[i].size()-1];
                p2 = line2->line_vector;
            }
            double theta = std::acosf((p1.x * p2.x + p1.y * p2.y) / (std::sqrt(p1.x * p1.x + p1.y * p1.y) * std::sqrt(p2.x * p2.x + p2.y * p2.y))) * 180 / CV_PI;
            pieces.pieces_lines[i][j].angle = theta;
            
            
            if(std::abs(theta - 90.0) < 2.0f){
                cv::line(warped, pieces.pieces_lines[i][j].start_point, pieces.pieces_lines[i][j].end_point, cv::Scalar(0,255,0), 3);
                cv::line(warped, line2->start_point, line2->end_point, cv::Scalar(0,255 ,0), 3);
                cv::imwrite("a4_warped.png", warped);
                std::cout << "Contour " << i << ", Line " << j << ": Angle = " << theta << " length= " << pieces.pieces_lines[i][j].length << std::endl;
                pieces.pieces_lines[i][j].leg_right_angle = true;
                line2->leg_right_angle = true;
                index.push_back(std::make_pair(i, j));
                angle_index[aindex] = theta;
                aindex++;
            }
        }
    }

    //寻找最长直角边，旋转到水平位置
    for(size_t i = 0; i < pieces.pieces_lines.size(); i++) {
        for (size_t j = 0; j < pieces.pieces_lines[i].size(); j++) {
            line_data* line1 = &pieces.pieces_lines[i][j];
            line_data* line2 = nullptr;
            if(j != 0) {
                line2 = &pieces.pieces_lines[i][j-1];
            }
            else {
                line2 = &pieces.pieces_lines[i][pieces.pieces_lines[i].size()-1];
            }
            if(line1->leg_right_angle && std::abs(line1->length - 100*scale) < 10.0f){ 
                if(line1->start_point == line2->end_point || line1->start_point == line2->start_point){
                    circle(lineImg, line1->start_point, 5, cv::Scalar(255), -1);
                    turn_angle(line1, 1);
                    for(size_t n = 0; n < pieces.pieces_lines[i].size(); n++){
                        if(n != j){
                            pieces.pieces_lines[i][n].turn_angle = line1->turn_angle;
                        }
                    }
                    Translate(pieces, cv::Point2f(640, 200) + (line1->start_point - pieces.center_gravity[i]), pieces.pieces_lines[i][j].turn_angle, i);
                    std::cout << "line1->start_point" << std::endl;
                }
                else{
                    turn_angle(line1, 2);
                    circle(lineImg, line1->end_point, 5, cv::Scalar(255), -1);
                    for(size_t n = 0; n < pieces.pieces_lines[i].size(); n++){
                        if(n != j){
                            pieces.pieces_lines[i][n].turn_angle = line1->turn_angle;
                        }
                    }
                    Translate(pieces, cv::Point2f(640, 200) + (line1->end_point - pieces.center_gravity[i]), pieces.pieces_lines[i][j].turn_angle, i);
                    std::cout << "line1->end_point" << std::endl;
                }
                
                circle(lineImg, cv::Point2f(640, 200), 3, cv::Scalar(255), -1);
                std::cout<< "line1->length: " << line1->length << " line_turn_angle: " << line1->turn_angle << std::endl;
                
                for (size_t k = 0; k < pieces.pieces_point[i].size(); k++) {
                    cv::circle(lineImg, pieces.pieces_point[i][k], 2, cv::Scalar(255), -1);
                    cv::line(lineImg, pieces.pieces_point[i][k], pieces.pieces_point[i][(k+1) % pieces.pieces_point[i].size()], cv::Scalar(255), 1);
                    cv::imwrite("lines.png", lineImg);
                }
                break;
            }
            
            
        }
    }
    

    cv::imwrite("lines.png", lineImg);
    cv::imwrite("gray.png", gray);
    cv::imwrite("edged.png", edged);
    cv::imwrite("binary.png", binary);
    // cv::imwrite("wrapped_line.png", wrapped_line);
}


void Translate(piece_data& pieces, cv::Point2f target_point, double theta, int index) {
    cv::Point2f origin_g_point = pieces.center_gravity[index];
    pieces.center_gravity[index] = target_point;
    theta = -theta * CV_PI / 180.0; // Convert degrees to radians
    for(size_t j = 0; j < pieces.pieces_point[index].size(); j++) {
        
        pieces.pieces_point[index][j].x = pieces.center_gravity[index].x + pieces.gra_p_vector[index][j].x * std::cos(theta) + pieces.gra_p_vector[index][j].y * std::sin(theta);
        pieces.pieces_point[index][j].y = pieces.center_gravity[index].y - pieces.gra_p_vector[index][j].x * std::sin(theta) + pieces.gra_p_vector[index][j].y * std::cos(theta);
        
        
    }
    
}

void turn_angle(line_data* line, int index) {
    cv::Point2f vector_x = cv::Point2f(1, 0);
    cv::Point2f vector_line;
    if(index == 1){
        vector_line = line->end_point - line->start_point;
    }
    else{
        
        vector_line = line->start_point - line->end_point;
    }
    
    std::cout << "vector_line: (" << vector_line.x << ", " << vector_line.y << ")" << std::endl;
    double theta = std::acosf((vector_x.x * vector_line.x + vector_x.y * vector_line.y)/(std::sqrt(vector_x.x * vector_x.x + vector_x.y * vector_x.y) * std::sqrt(vector_line.x * vector_line.x + vector_line.y * vector_line.y))) * 180 / CV_PI;
    if(vector_line.y < 0){
        theta = theta - 180.0;
    }
    else{
        theta = 180.0 - theta;
    }

    line->turn_angle = theta;
}