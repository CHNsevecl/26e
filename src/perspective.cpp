#include "perspective.hpp"
#include "piece_process.hpp"

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
    
    cv::Canny(gray, edged, 100, 150);
    
    cv::imwrite("edged_origin.png", edged);

    // 2. 找轮廓
    vector<vector<Point>> contours;
    findContours(edged, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 3. 筛选面积最大的四边形
    vector<Point2f> srcPts;
    double maxArea = 0;

    for (auto& c : contours) {
        double area = contourArea(c);
        std::cout << "轮廓面积: " << area << std::endl;
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
    std::vector<std::vector<struct line_data>> lines;
    cv::Mat lineImg = cv::Mat::zeros(cv::Size(papers_width*scale, papers_height*scale), CV_8UC1);
    piece_data pieces = line_process(warped, gray, blurred, edged, binary, lineImg, lines);
    pieces.pieces_lines = lines;
    pieces.turn_angle.resize(pieces.pieces_point.size(), 0.0);
    pieces.pieces_lines_right_point.resize(pieces.pieces_point.size());
    std::vector<size_t> finished_contour_index;
    finished_contour_index.reserve(pieces.pieces_point.size());

    for(size_t i = 0; i < pieces.pieces_lines_right_point.size(); i++) {
        pieces.pieces_lines_right_point[i].resize(pieces.pieces_lines[i].size(), false);
    }

    //计算重心
    gravity_calculation(pieces, lineImg);

    //边数据
    rect_length rects_length = {std::make_pair(cv::Point2f(240, 240), cv::Point2f(240 + 100.0*scale, 240)), // up
        std::make_pair(cv::Point2f(240, 240 + 60.0*scale), cv::Point2f(240 + 100.0*scale, 240 + 60.0*scale)), // down
        std::make_pair(cv::Point2f(240, 240), cv::Point2f(240, 240 + 60.0*scale)), // left
        std::make_pair(cv::Point2f(240 + 100.0*scale, 240), cv::Point2f(240 + 100.0*scale, 240 + 60.0*scale)), // right
        100.0*scale, // up_length
        100.0*scale, // down_length
        60.0*scale, // left_length
        60.0*scale // right_length
    };
    rect_position final_position = {rects_length.up.first, rects_length.right.first, rects_length.down.first, rects_length.right.second};
    cv::circle(lineImg, final_position.upper_left, 5, cv::Scalar(255), -1);
    cv::circle(lineImg, final_position.upper_right, 5, cv::Scalar(255), -1);
    cv::circle(lineImg, final_position.down_left, 5, cv::Scalar(255), -1);
    cv::circle(lineImg, final_position.down_right, 5, cv::Scalar(255), -1);

    //计算重心到点的向量0
    for(size_t i = 0; i < pieces.pieces_point.size(); i++) {
        for(size_t j = 0; j < pieces.pieces_point[i].size(); j++) {
            cv::Point2f vector = pieces.pieces_point[i][j] - pieces.center_gravity[i];
            pieces.gra_p_vector[i].push_back(vector);
            cv::circle(lineImg, pieces.pieces_point[i][j], 2, cv::Scalar(255), -1);
            cv::line(lineImg, pieces.center_gravity[i], pieces.center_gravity[i] + vector, cv::Scalar(255), 1);
            cv::imwrite("lines.png", lineImg);
        }   
    }
    
    //与同轮廓上一个索引的线的夹角，判断是否为直角边
    std::vector<std::pair<double, double>> index;
    double angle_index[100];
    int aindex = 0;
    leg_right_judgment(pieces, warped, index, angle_index, aindex);
    for(size_t i = 0; i < pieces.pieces_point.size(); i++) {
        for(size_t j = 0; j < pieces.pieces_point[i].size(); j++) {
            if(pieces.pieces_lines_right_point[i][j] == true){
                cv::line(lineImg, pieces.pieces_point[i][j], pieces.pieces_point[i][pieces.pieces_lines[i][j].start_point], cv::Scalar(255,0,0), 3);
                cv::circle(lineImg, pieces.pieces_point[i][j], 10, cv::Scalar(255), -1);
            }
        }
    }
    cv::imwrite("lines.png", lineImg);
    
    //余下的边长
    rect_length remain_length = rects_length;
    
    //计算拼图
    double max_right_contour = 0.0;
    size_t max_right_contour_index = 0;
    Traverse_Area(pieces, max_right_contour, max_right_contour_index);
    
    for(size_t k = 0; k < pieces.pieces_point[max_right_contour_index].size(); k++) {
        if( pieces.pieces_lines_right_point[max_right_contour_index][k] == false) {continue;}
        double max_right_length = 0.0;

        if(!l60_detect(pieces, max_right_contour_index, k)){
            if(!l100_detect(pieces, max_right_contour_index, k)){
                if(std::abs(pieces.pieces_lines[max_right_contour_index][k].length - 100*scale) < 10.0f && pieces.pieces_lines[max_right_contour_index][k].line_right_angle == true){
                    pieces.turn_angle[max_right_contour_index] = turn_angle(pieces, &pieces.pieces_lines[max_right_contour_index][k], max_right_contour_index);
                }
                if(std::abs(pieces.pieces_lines[max_right_contour_index][pieces.pieces_lines[max_right_contour_index][k].last_line_index].length - 100*scale) < 10.0f && pieces.pieces_lines[max_right_contour_index][pieces.pieces_lines[max_right_contour_index][k].last_line_index].line_right_angle == true){
                    
                    pieces.turn_angle[max_right_contour_index] = turn_angle(pieces, &pieces.pieces_lines[max_right_contour_index][k], max_right_contour_index);
                }
            }
        }
        finished_contour_index.push_back(max_right_contour_index);
        //还需要增加非60、100的边长检测，先不写

        cv::Point start_point = put_position(pieces, final_position, rects_length, max_right_contour_index, k);
   
        Translate(pieces, start_point, pieces.pieces_point[max_right_contour_index][k], pieces.turn_angle[max_right_contour_index], max_right_contour_index);

        for(size_t i = 0; i < pieces.pieces_point.size(); i++) {
            for (size_t j = 0; j < pieces.pieces_point[i].size(); j++) {
                cv::circle(lineImg, pieces.pieces_point[i][j], 2, cv::Scalar(255), -1);
                cv::line(lineImg, pieces.pieces_point[i][j], pieces.pieces_point[i][(j+1) % pieces.pieces_point[i].size()], cv::Scalar(255), 1);
                cv::imwrite("lines.png", lineImg);
            }
        }
    
    }

    //第一条边完成


    

    if(remain_length.left_length != rects_length.left_length && rects_length.left_length != 0.0f){
        remain_length = rects_length;
        for (size_t i = 0; i < pieces.pieces_point.size(); i++) {
            for (size_t j = 0; j < pieces.pieces_point[i].size(); j++) {
                if(pieces.pieces_lines_right_point[i][j] == true){    
                    if(std::abs(rects_length.down_length - pieces.pieces_lines[i][j].length) < 5.0f) {
                        cv::line(warped, pieces.pieces_point[i][pieces.pieces_lines[i][j].start_point], pieces.pieces_point[i][pieces.pieces_lines[i][j].end_point], cv::Scalar(0, 0, 255), 3);
                        cv::line(warped, pieces.pieces_point[i][pieces.pieces_lines[i][pieces.pieces_lines[i][j].last_line_index].start_point], pieces.pieces_point[i][pieces.pieces_lines[i][pieces.pieces_lines[i][j].last_line_index].end_point], cv::Scalar(0, 0, 255), 3);
                        cv::imwrite("a4_warped.png", warped);

                        piece_data pieces_temp = pieces;

                        pieces_temp.turn_angle[i] = turn_angle(pieces_temp, &pieces_temp.pieces_lines[i][j], i);

                        Translate(pieces_temp, final_position.upper_left, pieces_temp.pieces_point[i][j], pieces_temp.turn_angle[i], i);

                        //检测是否超出范围
                        bool re_turn = false;
                        for (size_t m = 0; m < pieces_temp.pieces_point[i].size(); m++) {
                            if(pieces_temp.pieces_point[i][m].x - final_position.down_right.x <  -1.0f || final_position.down_left.x - pieces_temp.pieces_point[i][m].x < -1.0f ||  final_position.down_left.y - pieces_temp.pieces_point[i][m].y < -1.0f || pieces_temp.pieces_point[i][m].y - final_position.upper_left.y < -1.0f){
                                re_turn = true;
                                pieces_temp = pieces;
                                break;
                            }
                        }
                        if(re_turn){
                            pieces_temp.turn_angle[i] = -(180.0-turn_angle(pieces_temp, &pieces_temp.pieces_lines[i][j], i));

                            Translate(pieces_temp, final_position.upper_left, pieces_temp.pieces_point[i][j], pieces_temp.turn_angle[i], i);
                            re_turn = false;
                        }


                        for (size_t m = 0; m < pieces_temp.pieces_point[i].size(); m++) {
                            if(pieces_temp.pieces_point[i][m].x - final_position.down_right.x <  -1.0f || final_position.down_left.x - pieces_temp.pieces_point[i][m].x < -1.0f ||  final_position.down_left.y - pieces_temp.pieces_point[i][m].y < -1.0f || pieces_temp.pieces_point[i][m].y - final_position.upper_left.y < -1.0f){
                                re_turn = true;
                                pieces_temp = pieces;
                                break;
                            }
                        }

                        if(re_turn){
                            pieces_temp.turn_angle[i] = turn_angle(pieces_temp, &pieces_temp.pieces_lines[i][pieces_temp.pieces_lines[i][j].last_line_index], i);

                            Translate(pieces_temp, final_position.upper_left, pieces_temp.pieces_point[i][j], pieces_temp.turn_angle[i], i);
                            re_turn = false;
                        }


                        for (size_t m = 0; m < pieces_temp.pieces_point[i].size(); m++) {
                            if(pieces_temp.pieces_point[i][m].x - final_position.down_right.x <  -1.0f || final_position.down_left.x - pieces_temp.pieces_point[i][m].x < -1.0f ||  final_position.down_left.y - pieces_temp.pieces_point[i][m].y < -1.0f || pieces_temp.pieces_point[i][m].y - final_position.upper_left.y < -1.0f){
                                re_turn = true;
                                pieces_temp = pieces;
                                break;
                            }
                        }
                        if(re_turn){
                            pieces_temp.turn_angle[i] = -(180.0-turn_angle(pieces_temp, &pieces_temp.pieces_lines[i][pieces_temp.pieces_lines[i][j].last_line_index], i));

                            Translate(pieces_temp, final_position.upper_left, pieces_temp.pieces_point[i][j], pieces_temp.turn_angle[i], i);
                            re_turn = false;
                        }

                        bool overlap = false;
                        for(size_t n = 0; n < finished_contour_index.size(); n++) {
                            
                            if(finished_contour_index[n] == i)  continue;
                            
                            if(polygonsOverlap(pieces_temp.pieces_point[finished_contour_index[n]], pieces_temp.pieces_point[i])){
                                overlap = true;
                                break;
                            }
                        }
                        
                        if(overlap){
                            std::cout << "拼图重叠，无法放置" << std::endl;
                            continue;
                        }

                        //检测邻边向量是否对应
                        bool vector_overlap = areLinesParallel(finished_contour_index, pieces_temp, i);
                        
                        if(!vector_overlap){
                            std::cout << "拼图边向量不匹配，无法放置" << std::endl;
                            continue;
                        }

                        pieces.turn_angle[i] = pieces_temp.turn_angle[i];
                        put_position(pieces, final_position, rects_length, i, j);
                        pieces = pieces_temp;
                        
                        
                        rects_length.down_length = rects_length.down_length - pieces.pieces_lines[i][j].length;
                        if(rects_length.down_length < 5.0f) rects_length.down_length = 0.0f;

                        for(size_t n = 0; n < pieces_temp.pieces_point.size(); n++) {
                            for (size_t k = 0; k < pieces_temp.pieces_point[n].size(); k++) {
                                cv::circle(lineImg, pieces_temp.pieces_point[n][k], 2, cv::Scalar(255), -1);
                                cv::line(lineImg, pieces_temp.pieces_point[n][k], pieces_temp.pieces_point[n][(k+1) % pieces_temp.pieces_point[n].size()], cv::Scalar(255), 1);
                                
                            }
                        }
                        cv::imwrite("lines.png", lineImg);
                    }

                    


                }
            }
        }
    }
    if(remain_length.down_length != rects_length.down_length && rects_length.down_length != 0.0f){
        remain_length = rects_length;
        for (size_t i = 0; i < pieces.pieces_point.size(); i++) {
            for (size_t j = 0; j < pieces.pieces_point[i].size(); j++) {
                if(pieces.pieces_lines_right_point[i][j] == true){    
                    if(std::abs(rects_length.down_length - pieces.pieces_lines[i][j].length) < 5.0f) {
                        cv::line(warped, pieces.pieces_point[i][pieces.pieces_lines[i][j].start_point], pieces.pieces_point[i][pieces.pieces_lines[i][j].end_point], cv::Scalar(0, 0, 255), 3);
                        cv::line(warped, pieces.pieces_point[i][pieces.pieces_lines[i][pieces.pieces_lines[i][j].last_line_index].start_point], pieces.pieces_point[i][pieces.pieces_lines[i][pieces.pieces_lines[i][j].last_line_index].end_point], cv::Scalar(0, 0, 255), 3);
                        cv::imwrite("a4_warped.png", warped);

                        piece_data pieces_temp = pieces;

                        pieces_temp.turn_angle[i] = turn_angle(pieces_temp, &pieces_temp.pieces_lines[i][j], i);

                        Translate(pieces_temp, final_position.down_right, pieces_temp.pieces_point[i][j], pieces_temp.turn_angle[i], i);

                        //检测是否超出范围
                        bool re_turn = false;
                        for (size_t m = 0; m < pieces_temp.pieces_point[i].size(); m++) {
                            if(pieces_temp.pieces_point[i][m].x - final_position.down_right.x <  -1.0f || final_position.down_left.x - pieces_temp.pieces_point[i][m].x < -1.0f ||  final_position.down_left.y - pieces_temp.pieces_point[i][m].y < -1.0f || pieces_temp.pieces_point[i][m].y - final_position.upper_left.y < -1.0f){
                                re_turn = true;
                                pieces_temp = pieces;
                                break;
                            }
                        }
                        if(re_turn){
                            pieces_temp.turn_angle[i] = -(180.0-turn_angle(pieces_temp, &pieces_temp.pieces_lines[i][j], i));

                            Translate(pieces_temp, final_position.down_right, pieces_temp.pieces_point[i][j], pieces_temp.turn_angle[i], i);
                            re_turn = false;
                        }


                        for (size_t m = 0; m < pieces_temp.pieces_point[i].size(); m++) {
                            if(pieces_temp.pieces_point[i][m].x - final_position.down_right.x <  -1.0f || final_position.down_left.x - pieces_temp.pieces_point[i][m].x < -1.0f ||  final_position.down_left.y - pieces_temp.pieces_point[i][m].y < -1.0f || pieces_temp.pieces_point[i][m].y - final_position.upper_left.y < -1.0f){
                                re_turn = true;
                                pieces_temp = pieces;
                                break;
                            }
                        }

                        if(re_turn){
                            pieces_temp.turn_angle[i] = turn_angle(pieces_temp, &pieces_temp.pieces_lines[i][pieces_temp.pieces_lines[i][j].last_line_index], i);

                            Translate(pieces_temp, final_position.down_right, pieces_temp.pieces_point[i][j], pieces_temp.turn_angle[i], i);
                            re_turn = false;
                        }



                        re_turn = areOutofarea(pieces_temp, final_position, i);
                        if(re_turn){
                            pieces_temp = pieces;
                            pieces_temp.turn_angle[i] = -(180.0-turn_angle(pieces_temp, &pieces_temp.pieces_lines[i][pieces_temp.pieces_lines[i][j].last_line_index], i));

                            Translate(pieces_temp, final_position.down_right, pieces_temp.pieces_point[i][j], pieces_temp.turn_angle[i], i);
                            re_turn = false;
                        }

                        bool overlap = false;
                        for(size_t n = 0; n < finished_contour_index.size(); n++) {
                            
                            if(finished_contour_index[n] == i)  continue;
                            
                            if(polygonsOverlap(pieces_temp.pieces_point[finished_contour_index[n]], pieces_temp.pieces_point[i])){
                                overlap = true;
                                break;
                            }
                        }
                        
                        if(overlap){
                            std::cout << "拼图重叠，无法放置" << std::endl;
                            continue;
                        }

                        //检测邻边向量是否对应
                        bool vector_overlap = areLinesParallel(finished_contour_index, pieces_temp, i);
                        
                        
                        if(!vector_overlap){
                            std::cout << "拼图边向量不匹配，无法放置" << std::endl;
                            continue;
                        }

                        pieces.turn_angle[i] = pieces_temp.turn_angle[i];//用put_position之前，必须确保pieces未修改且保有旋转角度
                        
                        
                        for (size_t k = 0; k < pieces_temp.pieces_point[i].size(); k++) {
                            cv::circle(lineImg, pieces_temp.pieces_point[i][k], 2, cv::Scalar(255), -1);
                            cv::line(lineImg, pieces_temp.pieces_point[i][k], pieces_temp.pieces_point[i][(k+1) % pieces_temp.pieces_point[i].size()], cv::Scalar(255), 1);
                        }
                        cv::imwrite("lines.png", lineImg);

                        put_position(pieces, final_position, rects_length, i, j);
                        
                        pieces = pieces_temp;
                        finished_contour_index.push_back(i);

                        if(rects_length.down_length < 5.0f) rects_length.down_length = 0.0f;

                        
                        //明日：将检测内容写为函数，检测是否超出范围，检测是否重叠，检测邻边向量是否对应
                    }

                    


                }
            }
        }

    }
    
   
    // put_position(pieces, final_position, rects_length, next_contour_index, next_line_index);
    //对于先放在右下角的拼图，这三个足够了
    //未考虑需要多个拼图完善底边、或底边不是直角的情况
    
    
    //第二条边完成

    // std::cout << remain_length.right_length<< std::endl;
    
    
    if(remain_length.right_length != rects_length.right_length && rects_length.right_length != 0.0f){
        
        remain_length = rects_length;

        //优化下部分代码
        for (size_t i = 0; i < pieces.pieces_point.size(); i++) {
            if(std::find(finished_contour_index.begin(), finished_contour_index.end(), i) != finished_contour_index.end()) {
                continue;
            }
            for (size_t j = 0; j < pieces.pieces_point[i].size(); j++) {
                 
                
                    if(std::abs(rects_length.right_length - pieces.pieces_lines[i][j].length) < 15.0f && pieces.pieces_lines[i][j].line_right_angle == true) {
                        //*+
                        
                    }
                    else if(pieces.pieces_lines[i][j].line_right_angle == true){
                        for(size_t k = 0; k < pieces.pieces_point.size(); k++) {
                            if((std::find(finished_contour_index.begin(), finished_contour_index.end(), k) != finished_contour_index.end() || i == k)) {
                                continue;
                            }
                            for(size_t m = 0; m < pieces.pieces_point[k].size(); m++) {
                                if(!(std::abs(rects_length.right_length - pieces.pieces_lines[k][m].length - pieces.pieces_lines[i][j].length) < 15.0f)) {
                                    continue;
                                }
                                std::cout << "找到匹配的拼图边长组合" << std::endl;

                                piece_data pieces_temp = pieces;


                                pieces_temp.turn_angle[k] = turn_angle(pieces_temp, &pieces_temp.pieces_lines[k][m], k) - 90.0f;
                                Translate(pieces_temp, rects_length.right.second, pieces_temp.pieces_point[k][m], pieces_temp.turn_angle[k], k);

                                bool re_turn = areOutofarea(pieces_temp, final_position, k);
                                if(re_turn){ // 异常
                                    pieces_temp = pieces;
                                    pieces_temp.turn_angle[k] = -(180.0-turn_angle(pieces_temp, &pieces_temp.pieces_lines[k][m], k)) - 90.0f;
                                    Translate(pieces_temp, rects_length.right.second, pieces_temp.pieces_point[k][m], pieces_temp.turn_angle[k], k);
                                    re_turn = false;
                                }
                                finished_contour_index.push_back(k);

                                // bool ifoverlap = false;
                                // for(auto& index: finished_contour_index) {
                                //     if(polygonsOverlap(pieces_temp.pieces_point[index], pieces_temp.pieces_point[k])){
                                //         std::cout << "拼图重叠，无法放置" << std::endl;
                                //         ifoverlap = true;
                                //         break;
                                //     }
                                // }
                                // if(ifoverlap){
                                //     continue;
                                // }

                                pieces = pieces_temp;

                                //第三条边



                                pieces_temp.turn_angle[i] = turn_angle(pieces_temp, &pieces_temp.pieces_lines[i][(j+1)%pieces_temp.pieces_lines[i].size()], i);
                                pieces.turn_angle[i] = pieces_temp.turn_angle[i];
                                cv::Point2f target_point = put_position(pieces_temp, final_position, rects_length, i, j);
                                pieces_temp = pieces;


                                cv::circle(lineImg, pieces_temp.pieces_point[i][j], 15, cv::Scalar(255), -1);
                                cv::Point2f right_point;
                                if(pieces.pieces_lines_right_point[i][j] == true){
                                    right_point = pieces_temp.pieces_point[i][j];
                                }
                                else {
                                    right_point = pieces_temp.pieces_point[i][pieces_temp.pieces_lines[i][j].start_point];
                                }
                                
                                Translate(pieces_temp, final_position.upper_right, right_point, pieces_temp.turn_angle[i], i);

                                re_turn = areOutofarea(pieces_temp, final_position, i);
                                if(re_turn){
                                    pieces_temp = pieces;
                                    pieces_temp.turn_angle[i] = (-90+turn_angle(pieces_temp, &pieces_temp.pieces_lines[i][(j+1)%pieces_temp.pieces_lines[i].size()], i));
                                    Translate(pieces_temp, final_position.upper_right, right_point, pieces_temp.turn_angle[i], i);
                                    re_turn = false;
                                }

                                re_turn = areOutofarea(pieces_temp, final_position, i);
                                if(re_turn){
                                    pieces_temp = pieces;
                                    pieces_temp.turn_angle[i] = (-180.0+turn_angle(pieces_temp, &pieces_temp.pieces_lines[i][(j+1)%pieces_temp.pieces_lines[i].size()], i));
                                    Translate(pieces_temp, final_position.upper_right, right_point, pieces_temp.turn_angle[i], i);
                                    re_turn = false;
                                }

                                re_turn = areOutofarea(pieces_temp, final_position, i);
                                if(re_turn){
                                    pieces_temp = pieces;
                                    pieces_temp.turn_angle[i] = -270 + turn_angle(pieces_temp, &pieces_temp.pieces_lines[i][(j+1)%pieces_temp.pieces_lines[i].size()], i);
                                    Translate(pieces_temp, final_position.upper_right, right_point, pieces_temp.turn_angle[i], i);
                                    re_turn = false;
                                }

                                finished_contour_index.push_back(i);
                                pieces = pieces_temp;
                                // 第四条边

                            
                                

                                for(size_t n = 0; n < pieces_temp.pieces_point[k].size(); n++) {
                                    cv::circle(lineImg, pieces_temp.pieces_point[k][n], 2, cv::Scalar(255), -1);
                                    cv::line(lineImg, pieces_temp.pieces_point[k][n], pieces_temp.pieces_point[k][(n+1) % pieces_temp.pieces_point[k].size()], cv::Scalar(255), 1);
                                }
                                for(size_t n = 0; n < pieces_temp.pieces_point[i].size(); n++) {
                                    cv::circle(lineImg, pieces_temp.pieces_point[i][n], 2, cv::Scalar(255), -1);
                                    cv::line(lineImg, pieces_temp.pieces_point[i][n], pieces_temp.pieces_point[i][(n+1) % pieces_temp.pieces_point[i].size()], cv::Scalar(255), 1);
                                }
                                cv::imwrite("lines.png", lineImg);
                               
                            }
                        }
                    }
                // }
                
            }
        }
    }

    if(remain_length.down_length != rects_length.down_length && rects_length.down_length != 0.0f){
        remain_length = rects_length;
        for (size_t i = 0; i < pieces.pieces_point.size(); i++) {
            for (size_t j = 0; j < pieces.pieces_point[i].size(); j++) {
                if(pieces.pieces_lines_right_point[i][j] == true){    
                    if(std::abs(rects_length.right_length - pieces.pieces_lines[i][j].length) < 5.0f) {
                        //
                    }
                }
            }
        }
    }



    cv::imwrite("lines.png", lineImg);
    cv::imwrite("gray.png", gray);
    cv::imwrite("edged.png", edged);
    cv::imwrite("binary.png", binary);
}


//有一条边容易被误识别为直角