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
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(binary, binary, kernel);
    cv::Canny(binary, edged, 0, 1);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edged, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat lineImg = cv::Mat::zeros(cv::Size(papers_width*scale, papers_height*scale), CV_8UC1);

    std::vector<std::vector<struct line_data>> lines(contours.size());
    std::cout << "Contours found: " << contours.size() << std::endl;
    for (size_t i = 0; i < contours.size(); i++) {
        std::vector<cv::Point> approx;
        double peri = cv::arcLength(contours[i], true);
        cv::approxPolyDP(contours[i], approx, 0.01 * peri, true);
        
        lines[i].reserve(approx.size());
        for (size_t j = 0; j < approx.size(); j++) {
            cv::Point p1 = approx[j];
            cv::Point p2 = approx[(j + 1) % approx.size()];
            double length = cv::norm(p2 - p1);
            double angle = std::atan2(p2.y - p1.y, p2.x - p1.x) * 180 / CV_PI;
            line_data ld{length, 0.0, 0.0, p2 - p1, p2 ,p1, std::make_pair(-1, -1)};
            lines[i].push_back(ld);
            
            cv::line(lineImg, p1, p2, cv::Scalar(255), 1);

            cv::imwrite("Line Segment.png", lineImg);
        }
    }

    

    //与同轮廓上一个索引的线的夹角，判断是否为直角边
    std::vector<std::pair<double, double>> index;
    double angle_index[100];
    int aindex = 0;
    for(size_t i = 0; i < lines.size(); i++) {
        for(size_t j = 0; j < lines[i].size(); j++) {
            cv::Point p1 = lines[i][j].line_vector;
            cv::Point p2;
            if(j != 0) {
                p2 = lines[i][j-1].line_vector;
            }
            else {
                p2 = lines[i][lines[i].size()-1].line_vector;
            }
            double theta = std::acosf((p1.x * p2.x + p1.y * p2.y) / (std::sqrt(p1.x * p1.x + p1.y * p1.y) * std::sqrt(p2.x * p2.x + p2.y * p2.y))) * 180 / CV_PI;
            lines[i][j].angle = theta;
            if(std::abs(theta - 90.0) < 5.0f){
                index.push_back(std::make_pair(i, j));
                angle_index[aindex] = theta;
                aindex++;
            }
        }
    }

    cv::Mat wrapped_line = warped.clone();
    std::vector<std::pair<double, double>> index_copy = index;
    cv::rectangle(wrapped_line, cv::Point(200, 100), cv::Point(200+100*scale, 100+60*scale), cv::Scalar(255, 255, 255), 2);
    rect_length rect = {100*scale, 100*scale, 60*scale, 60*scale};

    for (size_t i = 0; i < lines.size(); i++) {
        for(size_t j = 0; j < lines[i].size(); j++) {
            bool continue_flag = false;
            for(std::pair<double, double> in : index){
                if(in.first == i && in.second == j){
                    continue_flag = true;
                    break;
                }
            }
            if(continue_flag) continue;
            double length = lines[i][j].length;
            for(size_t k = i+1; k < lines.size(); k++) {
                for(size_t l = 0; l < lines[k].size(); l++) {
                    double length2 = lines[k][l].length;
                    // Skip the same line
                    if(std::abs(length - length2) < 10.0f) {
                        lines[i][j].near_line = std::make_pair(k, l);
                        std::cout << "Line " << i << "-" << j << " is near Line " << k << "-" << l << std::endl;
                        cv::line(wrapped_line, lines[i][j].start_point, lines[i][j].end_point, cv::Scalar(255, 0, 0), 2);
                        cv::line(wrapped_line, lines[k][l].start_point, lines[k][l].end_point, cv::Scalar(255, 0, 0), 2);
                        cv::imwrite("wrapped_line.png", wrapped_line);
                        ;
                    }
                }
            }   
        }
    }


    //第一块拼图
    for(size_t i = 0; i < index.size();i++){
        line_data* line1 = &lines[index[i].first][index[i].second];
        line_data* line2 = nullptr;
        if(index[i].second != 0){
            line2 = &lines[index[i].first][index[i].second-1];
        }
        else{
            line2 = &lines[index[i].first][lines[index[i].first].size()-1];
        }
        std::cout << "Angle: " << angle_index[i] << std::endl;
        cv::line(wrapped_line, line1->start_point, line1->end_point, cv::Scalar(10*i, 0, 255), 2);
        cv::line(wrapped_line, line2->start_point, line2->end_point, cv::Scalar(10*i, 0, 255), 2);

        //尝试找到有长度为100mm的线段的拼图，先将其放在左上角
        if(std::abs(line1->length - 100*scale) <  10.0f){
            first_piece(wrapped_line, line1, line2, rect, index);
            index_copy.erase(index_copy.begin() + i);
        }


        if(std::abs(line2->length - 100*scale) <  10.0f){
            first_piece(wrapped_line, line2, line1, rect, index);
            index_copy.erase(index_copy.begin() + i);
        }

        cv::imwrite("wrapped_line.png", wrapped_line);

    }

    //第二块拼图
    if(!index.empty()){
        for(size_t count = 0; count < index.size(); count++){
            line_data* line1 = &lines[index[count].first][index[count].second];
            line_data* line2 = nullptr;
            if(index[count].second != 0){
                line2 = &lines[index[count].first][index[count].second-1];
            }
            else{
                line2 = &lines[index[count].first][lines[index[count].first].size()-1];
            }

            if(std::abs(line1->length - 60*scale) <  20.0f){
                cv::line(wrapped_line, line1->start_point, line1->end_point, cv::Scalar(0, 255, 0), 2);
                std::cout << "Line1 length: " << line1->length << std::endl;
                if(rect.left != 60*scale){
                    second_piece(wrapped_line, line1, line2, rect, index);
                    index_copy.erase(index_copy.begin() + count);
                }
            }
            else if(std::abs(line2->length - 60*scale) <  20.0f){
                cv::line(wrapped_line, line2->start_point, line2->end_point, cv::Scalar(0, 255, 0), 2);
                std::cout << "Line2 length: " << line2->length << std::endl;
                if(rect.left != 60*scale){
                    second_piece(wrapped_line, line2, line1, rect, index);
                    index_copy.erase(index_copy.begin() + count);
                }

            }

            cv::imwrite("wrapped_line.png", wrapped_line);
        }
    }

    cv::imwrite("lines.png", lineImg);
    cv::imwrite("gray.png", gray);
    cv::imwrite("edged.png", edged);
    cv::imwrite("binary.png", binary);
    cv::imwrite("wrapped_line.png", wrapped_line);
}


//尝试找到有长度为100mm的线段的拼图，先将其放在右上角
void first_piece(cv::Mat& wrapped_line, line_data* line1, line_data* line2, rect_length& rect, const std::vector<std::pair<double, double>>& index){
    cv::line(wrapped_line, line1->start_point, line1->end_point, cv::Scalar(0, 255, 0), 2);
    line1->length = 100*(scale);
    double angle1;

    cv::Point vector_timely, vector_line;
    if(line1->start_point == line2->start_point || line1->start_point == line2->end_point){
        cv::Point timely_point = line1->start_point;
        timely_point.x -= 400;
        vector_timely = line1->start_point - timely_point;
        vector_line = line1->start_point - line1->end_point;
        
        
    }
    else{
        cv::Point timely_point = line1->end_point;
        timely_point.x -= 400;
        vector_timely = line1->end_point - timely_point;
        vector_line = line1->end_point - line1->start_point;
        
    }

    angle1 = std::acosf((vector_timely.x * vector_line.x + vector_timely.y * vector_line.y) / (std::sqrt(vector_timely.x * vector_timely.x + vector_timely.y * vector_timely.y) * std::sqrt(vector_line.x * vector_line.x + vector_line.y * vector_line.y))) * 180 / CV_PI;
    cv::line(wrapped_line, cv::Point(200+100*scale, 100), cv::Point(200, 100), cv::Scalar(0, 255, 0), 2);
    cv::line(wrapped_line, cv::Point(200+100*scale, 100), cv::Point(200+100*scale, 100+line2->length), cv::Scalar(0, 255, 0), 2);
    rect.upper -= line1->length;
    rect.left -= line2->length;
    line1->turn_angle = angle1;
    line2->turn_angle = angle1;

    
    std::cout << "Line1 length: " << line1->length << std::endl;
    std::cout << "turn angle: " << angle1 << std::endl;
}

//第二块拼图，将其放在左下角
void second_piece(cv::Mat& wrapped_line, line_data* line1, line_data* line2, rect_length& rect, const std::vector<std::pair<double, double>>& index){
    cv::line(wrapped_line, line1->start_point, line1->end_point, cv::Scalar(0, 255, 0), 2);
    line1->length = 60*(scale);
    double angle1;

    cv::Point vector_timely, vector_line;
    if(line1->start_point == line2->start_point || line1->start_point == line2->end_point){
        cv::Point timely_point = line1->start_point;
        timely_point.x -= line1->length;
        vector_timely = line1->start_point - timely_point;
        vector_line = line1->start_point - line1->end_point;
        
        
    }
    else{
        cv::Point timely_point = line1->end_point;
        timely_point.x -= line1->length;
        vector_timely = line1->end_point - timely_point;
        vector_line = line1->end_point - line1->start_point;
        
    }

    angle1 = std::acosf((vector_timely.x * vector_line.x + vector_timely.y * vector_line.y) / (std::sqrt(vector_timely.x * vector_timely.x + vector_timely.y * vector_timely.y) * std::sqrt(vector_line.x * vector_line.x + vector_line.y * vector_line.y))) * 180 / CV_PI;
    cv::line(wrapped_line, cv::Point(200, 100), cv::Point(200, 100+line1->length), cv::Scalar(0, 255, 0), 2);
    cv::line(wrapped_line, cv::Point(200, 100+line1->length), cv::Point(200+line2->length, 100+line1->length), cv::Scalar(0, 255, 0), 2);
    rect.upper -= line1->length;
    rect.left -= line2->length;
    line1->turn_angle = angle1;
    line2->turn_angle = angle1;

    std::cout << "turn angle: " << angle1 << std::endl;
}