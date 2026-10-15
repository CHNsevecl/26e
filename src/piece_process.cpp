#include "piece_process.hpp"

void Translate(piece_data& pieces, cv::Point2f target_point,cv::Point2f leg_tight_point, double theta, int index){
    cv::Point2f origin_g_point = pieces.center_gravity[index];
    
    theta = theta * CV_PI / 180.0; // Convert degrees to radians
    size_t ind = 0;
    for(size_t j = 0; j < pieces.gra_p_vector[index].size(); j++) {
        if(std::abs(pieces.pieces_point[index][j].x - leg_tight_point.x) < 1.0f && std::abs(pieces.pieces_point[index][j].y - leg_tight_point.y) < 1.0f) {
            ind = j;
            break;
        } 
    }

    for(size_t j = 0; j < pieces.pieces_point[index].size(); j++) {
        cv::Point2f temp_vector;
        temp_vector.x = pieces.gra_p_vector[index][j].x * std::cos(theta) + pieces.gra_p_vector[index][j].y * std::sin(theta);
        temp_vector.y = - pieces.gra_p_vector[index][j].x * std::sin(theta) + pieces.gra_p_vector[index][j].y * std::cos(theta);
        pieces.gra_p_vector[index][j] = temp_vector;
    }


    pieces.center_gravity[index].x = target_point.x - pieces.gra_p_vector[index][ind].x;
    pieces.center_gravity[index].y = target_point.y - pieces.gra_p_vector[index][ind].y;

    for(size_t j = 0; j < pieces.pieces_point[index].size(); j++) {
        cv::Point2f temp_vector;
        temp_vector.x = pieces.center_gravity[index].x + pieces.gra_p_vector[index][j].x;
        temp_vector.y = pieces.center_gravity[index].y + pieces.gra_p_vector[index][j].y;
        pieces.pieces_point[index][j] = temp_vector;
    }

    for(size_t j = 0; j < pieces.pieces_lines[index].size(); j++) {
        pieces.pieces_lines[index][j].line_vector = pieces.pieces_point[index][pieces.pieces_lines[index][j].end_point] - pieces.pieces_point[index][pieces.pieces_lines[index][j].start_point];
    }
    
}

piece_data line_process(const cv::Mat& warped, cv::Mat& gray, cv::Mat& blurred, cv::Mat& edged, cv::Mat& binary ,cv::Mat& lineImg, std::vector<std::vector<struct line_data>>& lines){
    
    cv::cvtColor(warped, gray, cv::COLOR_BGR2GRAY);
    
    std::vector<cv::Point2f> pts;
   
    cv::GaussianBlur(gray, blurred, cv::Size(3, 3), 1.0);  // 核可适当加大
    // cv::medianBlur(gray, blurred, 3);
    cv::imwrite("blurred.png", blurred);
    cv::threshold(blurred, binary, 230, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::imwrite("origin_binary.png", binary);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1));
    cv::dilate(binary, binary, kernel);
    cv::Canny(binary, edged, 0, 10);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edged, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    lines.resize(contours.size());
    struct piece_data pieces;
    pieces.pieces_point.resize(contours.size());
    pieces.center_gravity.resize(contours.size());
    pieces.gra_p_vector.resize(pieces.pieces_point.size());

    std::cout << "Contours found: " << contours.size() << std::endl;
    for (size_t i = 0; i < contours.size(); i++) {
        std::vector<cv::Point> approx;
        double peri = cv::arcLength(contours[i], true);
        cv::approxPolyDP(contours[i], approx, 0.01 * peri, true);
        
        std::vector<cv::Point2f> pts;
        pts.reserve(approx.size());
        for (const auto& p : approx) {
            pts.emplace_back(static_cast<float>(p.x),
                             static_cast<float>(p.y));
        }

        // 7.3 亚像素精修（输入必须是灰度图 gray）
        cv::cornerSubPix(gray, pts,
                         cv::Size(5, 5),
                         cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS +
                                          cv::TermCriteria::MAX_ITER,
                                          30, 0.1));

        // 7.4 clamp 到图像范围内，防止画线越界
        for (auto& p : pts) {
            p.x = std::max(0.f, std::min(p.x, static_cast<float>(gray.cols - 1)));
            p.y = std::max(0.f, std::min(p.y, static_cast<float>(gray.rows - 1)));
        }

        // 7.5 用精修后的顶点生成线段
        lines[i].reserve(pts.size());
        pieces.pieces_point[i].reserve(pts.size());

        for (size_t j = 0; j < approx.size(); j++) {
            cv::Point2f p1 = approx[j];
            cv::Point2f p2 = approx[(j + 1) % approx.size()];
            double length = cv::norm(p2 - p1);
            double angle = std::atan2(p2.y - p1.y, p2.x - p1.x) * 180 / CV_PI;
            line_data ld{length, 0.0, p2 - p1, (j + 1) % approx.size() ,j, std::make_pair(-1, -1), false,0};
            lines[i].push_back(ld);
            pieces.pieces_point[i].push_back(p1);
            cv::line(lineImg, p1, p2, cv::Scalar(255), 1);

            cv::imwrite("Line Segment.png", lineImg);
        }
    }

    return pieces;
}

void gravity_calculation(piece_data& pieces, cv::Mat& lineImg) {
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
}

void leg_right_judgment(piece_data& pieces,const cv::Mat& warped, std::vector<std::pair<double, double>>& index, double* angle_index, int& aindex) { 
    for(size_t i = 0; i < pieces.pieces_lines.size(); i++) {
        for(size_t j = 0; j < pieces.pieces_lines[i].size(); j++) {
            line_data* line2 = nullptr;
            cv::Point p1 = pieces.pieces_lines[i][j].line_vector;
            cv::Point p2;
            if(j != 0) {
                line2 = &pieces.pieces_lines[i][j-1];
                pieces.pieces_lines[i][j].last_line_index = j-1;
                p2 = line2->line_vector;
            }
            else {
                line2 = &pieces.pieces_lines[i][pieces.pieces_lines[i].size()-1];
                pieces.pieces_lines[i][j].last_line_index = pieces.pieces_lines[i].size()-1;
                p2 = line2->line_vector;
            }
            double theta = std::acosf((p1.x * p2.x + p1.y * p2.y) / (std::sqrt(p1.x * p1.x + p1.y * p1.y) * std::sqrt(p2.x * p2.x + p2.y * p2.y))) * 180 / CV_PI;
            pieces.pieces_lines[i][j].angle = theta;
            
            
            
            if(std::abs(theta - 90.0) < 3.0f){
                cv::line(warped, pieces.pieces_point[i][j], pieces.pieces_point[i][pieces.pieces_lines[i][j].start_point], cv::Scalar(255,0,0), 3);
                cv::line(warped, pieces.pieces_point[i][line2->start_point], pieces.pieces_point[i][line2->end_point], cv::Scalar(255,0 ,0), 3);
                cv::imwrite("a4_warped.png", warped);
                if(pieces.pieces_lines[i][j].start_point == line2->end_point || pieces.pieces_lines[i][j].start_point == line2->start_point){
                    for(size_t k = 0; k < pieces.pieces_point[i].size(); k++) {
                        if(pieces.pieces_point[i][k] == pieces.pieces_point[i][j]){
                            pieces.pieces_lines_right_point[i][j] = true;
                            break;
                        }
                    }
                }
                else{
                    for(size_t k = 0; k < pieces.pieces_point[i].size(); k++) {
                        if(pieces.pieces_point[i][k] == pieces.pieces_point[i][pieces.pieces_lines[i][j].end_point]){
                            pieces.pieces_lines_right_point[i][j] = true;
                            break;
                        }
                    }
                }
                pieces.pieces_lines[i][j].line_right_angle = true;
                line2->line_right_angle = true;
                index.push_back(std::make_pair(i, j));
                angle_index[aindex] = theta;
                aindex++;
            }
        }
    }
}

double polygonArea(const std::vector<cv::Point2f>& pts){ //计算面积
    int n = static_cast<int>(pts.size());
    if (n < 3) return 0.0f;

    double area = 0.0f;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;   // 下一个点，循环回到起点
        area += pts[i].x * pts[j].y - pts[j].x * pts[i].y;
    }
    return std::abs(area) * 0.5f;
}

cv::Point2f put_position(piece_data pieces_temp, rect_position& final_position, rect_length& rects_length, int max_right_contour_index, int k_index){

    Translate(pieces_temp, cv::Point2f(640, 480) , pieces_temp.pieces_point[max_right_contour_index][k_index], pieces_temp.turn_angle[max_right_contour_index], max_right_contour_index);
        

    cv::Point2f vector1 = -pieces_temp.pieces_point[max_right_contour_index][k_index] + pieces_temp.pieces_point[max_right_contour_index][(k_index+1) % pieces_temp.pieces_point[max_right_contour_index].size()];
    cv::Point2f vector2 = -pieces_temp.pieces_point[max_right_contour_index][k_index] + pieces_temp.pieces_point[max_right_contour_index][pieces_temp.pieces_lines[max_right_contour_index][k_index].last_line_index];


    cv::Point2f start_point;
    //存在直角边时
    //左下直角
    if((std::abs(vector1.x) < 5.0f && vector1.y < 0.0f && std::abs(vector2.y) < 5.0f && vector2.x > 0.0f) || (std::abs(vector2.x) < 5.0f && vector2.y < 0.0f && std::abs(vector1.y) < 5.0f && vector1.x > 0.0f)){
        start_point = final_position.down_left;
        if(std::abs(vector1.y) < 5.0f && vector1.x > 0.0f && std::abs(vector2.x) < 5.0f && vector2.y < 0.0f){
            rects_length.left.second.y -= std::abs(vector2.y);
            rects_length.down.first.x += std::abs(vector1.x);
        }
        else{
            rects_length.left.second.y -= std::abs(vector1.y);
            rects_length.down.first.x += std::abs(vector2.x);
        }

        rects_length.left_length = rects_length.left.second.y -  rects_length.left.first.y;
        rects_length.down_length = rects_length.down.second.x -  rects_length.down.first.x;
        if(std::abs(rects_length.left_length) < 5.0f){
            rects_length.left_length = 0.0f;
            rects_length.left.second = rects_length.left.first;
        }
        if(std::abs(rects_length.down_length) < 5.0f){
            rects_length.down_length = 0.0f;
            rects_length.down.second = rects_length.down.first;
        }
    }
    //右下直角
    else if((std::abs(vector1.x) < 5.0f && vector1.y < 0.0f && std::abs(vector2.y) < 5.0f && vector2.x < 0.0f) || (std::abs(vector2.x) < 5.0f && vector2.y < 0.0f && std::abs(vector1.y) < 5.0f && vector1.x < 0.0f)){
        start_point = final_position.down_right;
        if(std::abs(vector1.y) < 5.0f && vector1.x < 0.0f && std::abs(vector2.x) < 5.0f && vector2.y < 0.0f){
            rects_length.right.second.y -= std::abs(vector2.y);
            rects_length.down.second.x -= std::abs(vector1.x);
        }
        else{
            rects_length.right.second.y -= std::abs(vector1.y);
            rects_length.down.second.x -= std::abs(vector2.x);
        }

        rects_length.right_length = rects_length.right.second.y -  rects_length.right.first.y;
        rects_length.down_length = rects_length.down.second.x -  rects_length.down.first.x;
        if(std::abs(rects_length.right_length) < 5.0f){
            rects_length.right_length = 0.0f;
            rects_length.right.second = rects_length.right.first;
        }
        if(std::abs(rects_length.down_length) < 5.0f){
            rects_length.down_length = 0.0f;
            rects_length.down.second = rects_length.down.first;
        }
    }
    else if((std::abs(vector1.x) < 5.0f && vector1.y > 0.0f && std::abs(vector2.y) < 5.0f && vector2.x < 0.0f) || (std::abs(vector2.x) < 5.0f && vector2.y > 0.0f && std::abs(vector1.y) < 5.0f && vector1.x < 0.0f)){
        start_point = final_position.upper_right;
        if(std::abs(vector1.y) < 5.0f && vector1.x < 0.0f && std::abs(vector2.x) < 5.0f && vector2.y > 0.0f){
            rects_length.right.first.y += std::abs(vector2.y);
            rects_length.up.second.x -= std::abs(vector1.x);
        }
        else{
            rects_length.right.first.y += std::abs(vector1.y);
            rects_length.up.second.x -= std::abs(vector2.x);
        }

        rects_length.right_length = rects_length.right.second.y -  rects_length.right.first.y;
        rects_length.up_length = rects_length.up.second.x -  rects_length.up.first.x;
        if(std::abs(rects_length.right_length) < 5.0f){
            rects_length.right_length = 0.0f;
            rects_length.right.second = rects_length.right.first;
        }
        if(std::abs(rects_length.up_length) < 5.0f){
            rects_length.up_length = 0.0f;
            rects_length.up.second = rects_length.up.first;
        }
    }
    else if((std::abs(vector1.x) < 5.0f && vector1.y > 0.0f && std::abs(vector2.y) < 5.0f && vector2.x > 0.0f) || (std::abs(vector2.x) < 5.0f && vector2.y > 0.0f && std::abs(vector1.y) < 5.0f && vector1.x > 0.0f)){
        start_point = final_position.upper_left;
        if(std::abs(vector1.y) < 5.0f && vector1.x > 0.0f && std::abs(vector2.x) < 5.0f && vector2.y > 0.0f){
            rects_length.left.first.y += std::abs(vector2.y);
            rects_length.up.first.x += std::abs(vector1.x);
        }
        else{
            rects_length.left.first.y += std::abs(vector1.y);
            rects_length.up.first.x += std::abs(vector2.x);
        }

        rects_length.left_length = rects_length.left.second.y -  rects_length.left.first.y;
        rects_length.up_length = rects_length.up.second.x -  rects_length.up.first.x;
        if(std::abs(rects_length.left_length) < 5.0f){
            rects_length.left_length = 0.0f;
            rects_length.left.second = rects_length.left.first;
        }
        if(std::abs(rects_length.up_length) < 5.0f){
            rects_length.up_length = 0.0f;
            rects_length.up.second = rects_length.up.first;
        }
    }
    

    return start_point;
}

bool l60_detect(piece_data& pieces,size_t& contour_index, size_t& line_index ){
    if(std::abs(pieces.pieces_lines[contour_index][line_index].length - 60*scale) < 10.0f && pieces.pieces_lines[contour_index][line_index].line_right_angle == true){
        pieces.turn_angle[contour_index] = turn_angle(pieces, &pieces.pieces_lines[contour_index][line_index], contour_index);
        return true;
    }
    if(std::abs(pieces.pieces_lines[contour_index][pieces.pieces_lines[contour_index][line_index].last_line_index].length - 60*scale) < 10.0f && pieces.pieces_lines[contour_index][pieces.pieces_lines[contour_index][line_index].last_line_index].line_right_angle == true){
        
        pieces.turn_angle[contour_index] = turn_angle(pieces, &pieces.pieces_lines[contour_index][line_index], contour_index);
        return true;
    }
    return false;
}

//未测试
bool l100_detect(piece_data& pieces,size_t& contour_index, size_t& line_index ){
    if(std::abs(pieces.pieces_lines[contour_index][line_index].length - 100*scale) < 10.0f && pieces.pieces_lines[contour_index][line_index].line_right_angle == true){
        pieces.turn_angle[contour_index] = turn_angle(pieces, &pieces.pieces_lines[contour_index][line_index], contour_index);
        return true;
    }
    if(std::abs(pieces.pieces_lines[contour_index][pieces.pieces_lines[contour_index][line_index].last_line_index].length - 100*scale) < 10.0f && pieces.pieces_lines[contour_index][pieces.pieces_lines[contour_index][line_index].last_line_index].line_right_angle == true){
        
        pieces.turn_angle[contour_index] = turn_angle(pieces, &pieces.pieces_lines[contour_index][line_index], contour_index);
        return true;
    }
    return false;
}

bool Traverse_Area(piece_data& pieces, double& max_right_contour, size_t& max_right_contour_index){
    bool right_angle = false;
    for(size_t i = 0; i < pieces.pieces_point.size(); i++) {
        for (size_t j = 0; j < pieces.pieces_point[i].size(); j++) {
            if(pieces.pieces_lines_right_point[i][j]) {
                right_angle = true;
                break;
            }
        }
        if(right_angle){
            double area = polygonArea(pieces.pieces_point[i]);
            if(area > max_right_contour){
                max_right_contour = area;   
                max_right_contour_index = i;
            }
            continue;
        }
        right_angle = false;
    }
    max_right_contour = 0.0; 
    if(right_angle){
        return true;
    }
    else{
        return false;
    }
}

//旋转时，是以直角为原点的向量计算，计算该向量与x轴的夹角，返回旋转角度
//若直角在右侧，要注意方向改变
double turn_angle(piece_data& pieces, line_data* line, size_t contour_index) {
    cv::Point2f vector_x = cv::Point2f(1, 0);
    cv::Point2f vector_line;
    if(pieces.pieces_lines_right_point[contour_index][line->start_point] == true){
        vector_line = pieces.pieces_point[contour_index][line->end_point] - pieces.pieces_point[contour_index][line->start_point];
    }
    else{
        vector_line = pieces.pieces_point[contour_index][line->start_point] - pieces.pieces_point[contour_index][line->end_point];
    }

    double theta = std::acosf((vector_x.x * vector_line.x + vector_x.y * vector_line.y)/(std::sqrt(vector_x.x * vector_x.x + vector_x.y * vector_x.y) * std::sqrt(vector_line.x * vector_line.x + vector_line.y * vector_line.y))) * 180 / CV_PI;
    if(vector_line.y < 0){
        // theta = theta - 180.0;
        theta = -theta;
    }
    // else{
    //     theta = 180.0 - theta;
    // }

    return theta;
}


//
static bool pointDeepInsidePolygon(const cv::Point2f& pt,
                                   const std::vector<cv::Point2f>& poly,
                                   double tol)
{
    double d = cv::pointPolygonTest(poly, pt, true); // true: 返回有符号距离
    // d > 0 表示在内部，d 的绝对值是到最近边的距离
    return d > tol;
}

// 判断两条线段是否"真正"相交（交点不在容差端点附近）
static bool segmentsCrossStrict(const cv::Point2f& p1, const cv::Point2f& p2,
                                const cv::Point2f& q1, const cv::Point2f& q2,
                                double tol)
{
    auto cross = [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.x * b.y - a.y * b.x;
    };
    auto sub = [](const cv::Point2f& a, const cv::Point2f& b) {
        return cv::Point2f(a.x - b.x, a.y - b.y);
    };
    auto len = [](const cv::Point2f& v) {
        return std::sqrt(v.x * v.x + v.y * v.y);
    };

    cv::Point2f r = sub(p2, p1);
    cv::Point2f s = sub(q2, q1);
    float denom = cross(r, s);
    cv::Point2f qp = sub(q1, p1);

    if (std::abs(denom) < 1e-9f) {
        // 平行（含共线）→ 不算"真正交叉"
        return false;
    }

    float t = cross(qp, s) / denom;
    float u = cross(qp, r) / denom;

    // 必须在两条线段内部（不在端点附近）
    if (t < 0.f || t > 1.f || u < 0.f || u > 1.f)
        return false;

    // 计算交点到各端点的距离，判断是否在容差范围内
    cv::Point2f inter(p1.x + t * r.x, p1.y + t * r.y);

    // 如果交点距离任意端点 <= tol，视为"贴合"，不算真正交叉
    if (len(sub(inter, p1)) <= tol) return false;
    if (len(sub(inter, p2)) <= tol) return false;
    if (len(sub(inter, q1)) <= tol) return false;
    if (len(sub(inter, q2)) <= tol) return false;

    return true;
}

// 主函数：判断两个多边形是否重合（顶点/边在 tol 像素内的接触不算）
bool polygonsOverlap(const std::vector<cv::Point2f>& polyA,
                     const std::vector<cv::Point2f>& polyB,
                     double tol)
{
    if (polyA.size() < 3 || polyB.size() < 3)
        return false;

    size_t nA = polyA.size(), nB = polyB.size();

    // 1. 检查 A 的顶点是否深入 B 内部超过 tol
    for (const auto& p : polyA) {
        if (pointDeepInsidePolygon(p, polyB, tol))
            return true;
    }

    // 2. 检查 B 的顶点是否深入 A 内部超过 tol
    for (const auto& p : polyB) {
        if (pointDeepInsidePolygon(p, polyA, tol))
            return true;
    }

    // 3. 检查边是否"真正"交叉（交点不在容差端点附近）
    for (size_t i = 0; i < nA; ++i) {
        const cv::Point2f& a1 = polyA[i];
        const cv::Point2f& a2 = polyA[(i + 1) % nA];
        for (size_t j = 0; j < nB; ++j) {
            const cv::Point2f& b1 = polyB[j];
            const cv::Point2f& b2 = polyB[(j + 1) % nB];
            if (segmentsCrossStrict(a1, a2, b1, b2, tol))
                return true;
        }
    }

    return false;
}

bool areLinesParallel(std::vector<size_t> finished_contour_index, piece_data& pieces_temp, size_t contour_index) {
    for(auto& finished_index : finished_contour_index) {
        for (size_t k = 0; k < pieces_temp.pieces_point[finished_index].size(); k++) {
            if(pieces_temp.pieces_lines_right_point[finished_index][k] == true){
                continue;
            }
            cv::Point2f vector_x = pieces_temp.pieces_lines[finished_index][k].line_vector;
            for (size_t m = 0; m < pieces_temp.pieces_point[contour_index].size(); m++) {
                if(pieces_temp.pieces_lines_right_point[contour_index][m] == true){
                    continue;
                }
                cv::Point2f vector_line = pieces_temp.pieces_lines[contour_index][m].line_vector;
                
                double theta = std::acosf((vector_x.x * vector_line.x + vector_x.y * vector_line.y)/(std::sqrt(vector_x.x * vector_x.x + vector_x.y * vector_x.y) * std::sqrt(vector_line.x * vector_line.x + vector_line.y * vector_line.y))) * 180 / CV_PI;
                if((std::abs(theta) < 5.0f || std::abs(theta - 180.0f) < 5.0f)){
                    // Lines are parallel
                    if(cv::norm(pieces_temp.pieces_point[contour_index][m] - pieces_temp.pieces_point[finished_index][k]) < 10.0f || 
                        cv::norm(pieces_temp.pieces_point[contour_index][m] - pieces_temp.pieces_point[finished_index][pieces_temp.pieces_lines[finished_index][k].start_point]) < 10.0f||
                        cv::norm(pieces_temp.pieces_point[contour_index][pieces_temp.pieces_lines[contour_index][m].start_point] - pieces_temp.pieces_point[finished_index][k]) < 10.0f ||
                        cv::norm(pieces_temp.pieces_point[contour_index][pieces_temp.pieces_lines[contour_index][m].start_point] - pieces_temp.pieces_point[finished_index][pieces_temp.pieces_lines[finished_index][k].start_point]) < 10.0f){
                        
                        return true;
                    }
                }
            }
        }
    }
    
    return false;
}

bool areOutofarea(piece_data& pieces_temp, rect_position& final_position, size_t contour_index, double tol) {
    for (size_t m = 0; m < pieces_temp.pieces_point[contour_index].size(); m++) {
        if(final_position.down_right.x - pieces_temp.pieces_point[contour_index][m].x < -tol || final_position.down_left.x - pieces_temp.pieces_point[contour_index][m].x > tol ||  final_position.down_left.y - pieces_temp.pieces_point[contour_index][m].y < -tol || final_position.upper_left.y - pieces_temp.pieces_point[contour_index][m].y  > tol){
            return true;
            // pieces_temp = pieces;
            break;
        }
    }

    return false;
}