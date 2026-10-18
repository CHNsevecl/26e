#include "perspective.hpp"
#include "piece_process.hpp"
#include "puzzle_solver.hpp"

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

void task(const cv::Mat& warped) {
    if (warped.empty() || warped.type() != CV_8UC3) {
        std::cerr << "拼图失败：输入必须是非空 BGR 图像" << std::endl;
        return;
    }

    cv::Mat gray, blurred, edged, binary;
    std::vector<std::vector<line_data>> lines;
    cv::Mat recognition = cv::Mat::zeros(warped.size(), CV_8UC1);
    const piece_data pieces = line_process(warped, gray, blurred, edged, binary, recognition, lines);

    piece_data pieces_origin = pieces; // 复制一份用于后续处理

    // 保留原来的 100 mm × 60 mm 外框和显示位置。
    const cv::Rect2f target(240, 240, 100.0f * scale, 60.0f * scale);
    const PuzzleOptions options;
    const auto solve_start = cv::getTickCount();
    const PuzzleResult result = solve_puzzle(pieces.pieces_point, target, options);
    const double solve_time_ms = (cv::getTickCount() - solve_start) * 1000.0 / cv::getTickFrequency();
    double input_area = 0.0;
    int input_count = 0;
    for (const auto& polygon : pieces.pieces_point) {
        if (polygon.empty()) continue;
        ++input_count;
        input_area += std::abs(cv::contourArea(polygon));
    }

    // 最终画布仅包含确认成功的结果，不累积原位置和被否决的候选。
    cv::Mat final_image = cv::Mat::zeros(cv::Size(papers_width * scale, papers_height * scale), CV_8UC3);
    cv::rectangle(final_image, cv::Point2f(target.x, target.y),
                  cv::Point2f(target.x + target.width, target.y + target.height), cv::Scalar(120, 120, 120), 1);
    const std::vector<cv::Scalar> colors = {
        {80, 220, 255}, {100, 255, 100}, {255, 180, 100}, {220, 100, 255}
    };

    cv::FileStorage report("puzzle_result.yml", cv::FileStorage::WRITE);
    if (report.isOpened()) {
        report << "solved" << static_cast<int>(result.solved)
               << "search_limit_reached" << static_cast<int>(result.search_limit_reached)
               << "explored_nodes" << static_cast<double>(result.explored_nodes)
               << "solve_time_ms" << solve_time_ms
               << "input_piece_count" << input_count
               << "input_area_pixels" << input_area
               << "coverage" << result.coverage
               << "overlap_area_pixels" << result.overlap_area
               << "outside_area_pixels" << result.outside_area
               << "position_tolerance_pixels" << options.position_tolerance
               << "target" << target
               << "min_coverage" << options.min_coverage
               << "max_overlap_ratio" << options.max_overlap_ratio
               << "max_outside_ratio" << options.max_outside_ratio
               << "source_polygons" << "[";
        for (std::size_t i = 0; i < pieces.pieces_point.size(); ++i)
            if (!pieces.pieces_point[i].empty())
                report << "{" << "source_index" << static_cast<int>(i) << "points" << pieces.pieces_point[i] << "}";
        report << "]" << "placements" << "[";
    }
    for (const auto& placement : result.placements) {
        const auto color = colors[placement.source_index % colors.size()];
        for (std::size_t j = 0; j < placement.points.size(); ++j) {
            cv::line(final_image, placement.points[j], placement.points[(j + 1) % placement.points.size()], color, 1, cv::LINE_AA);
            cv::circle(final_image, placement.points[j], 2, color, cv::FILLED, cv::LINE_AA);
        }
        if (report.isOpened()) {
            report << "{" << "source_index" << static_cast<int>(placement.source_index)
                   << "rotation_degrees" << placement.rotation_radians * 180.0 / CV_PI
                   << "center_gravity" << placement.center_gravity
                   << "translation" << placement.translation << "points" << placement.points << "}";
        }
    }
    if (report.isOpened()) report << "]";
    if (!cv::imwrite("lines.png", final_image)) std::cerr << "无法保存 lines.png" << std::endl;

    if (result.solved) {
        std::cout << "拼图完成：" << result.placements.size() << " 块，覆盖率 "
                  << result.coverage * 100.0 << "% ，重叠面积 " << result.overlap_area
                  << " 像素²，框外面积 " << result.outside_area << " 像素²" << std::endl;
    } else {
        std::cerr << (result.search_limit_reached ? "搜索达到上限，尚未找到满足容差的拼法" : "未找到满足覆盖、重叠和边界约束的拼法")
                  << "；请检查识别轮廓和 puzzle_result.yml" << std::endl;
    }
}
