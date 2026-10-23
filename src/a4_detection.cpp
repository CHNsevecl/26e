#include "a4_detection.hpp"
#include "perspective.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

using namespace cv;

std::vector<Point2f> orderPoints(std::vector<Point2f> points) {
    if (points.size() != 4) return {};
    Point2f center;
    for (const auto& p : points) center += p * 0.25f;
    std::sort(points.begin(), points.end(), [center](Point2f a, Point2f b) {
        return std::atan2(a.y - center.y, a.x - center.x) < std::atan2(b.y - center.y, b.x - center.x);
    });
    const auto first = std::min_element(points.begin(), points.end(), [](Point2f a, Point2f b) {
        return a.x + a.y < b.x + b.y;
    });
    std::rotate(points.begin(), first, points.end());
    return points;
}

namespace {
struct Evidence {
    double score = 0;
    bool accepted = false;
};

Evidence evaluate(const Mat& gray, const std::vector<Point2f>& corners, double dark_threshold) {
    const double area_ratio = std::abs(contourArea(corners)) / gray.total();
    if (area_ratio < 0.06 || area_ratio > 0.95 || !isContourConvex(corners)) return {};
    std::array<double, 4> sides;
    for (int i = 0; i < 4; ++i) {
        const Point2d before = corners[(i + 3) % 4] - corners[i];
        const Point2d after = corners[(i + 1) % 4] - corners[i];
        sides[i] = norm(after);
        if (sides[i] < std::min(gray.rows, gray.cols) * 0.10 ||
            std::abs(before.dot(after)) > 0.75 * norm(before) * norm(after)) return {};
    }
    const double width = (sides[0] + sides[2]) / 2;
    const double height = (sides[1] + sides[3]) / 2;
    const double aspect = std::min(width, height) / std::max(width, height);
    if (aspect < 0.4) return {};

    const std::vector<Point2f> unit = {{0,0},{1,0},{1,1},{0,1}};
    const Mat map = getPerspectiveTransform(unit, corners);
    auto sample = [&](double x, double y, double& value) {
        const double z = map.at<double>(2,0)*x + map.at<double>(2,1)*y + map.at<double>(2,2);
        if (std::abs(z) < 1e-9) return false;
        const double px = (map.at<double>(0,0)*x + map.at<double>(0,1)*y + map.at<double>(0,2))/z;
        const double py = (map.at<double>(1,0)*x + map.at<double>(1,1)*y + map.at<double>(1,2))/z;
        if (!std::isfinite(px) || !std::isfinite(py) || px < 0 || py < 0 ||
            px >= gray.cols-1 || py >= gray.rows-1) return false;
        value = gray.at<uchar>(cvRound(py), cvRound(px));
        return true;
    };
    int dark = 0, total = 0;
    for (int y = 0; y < 32; ++y)
        for (int x = 0; x < 24; ++x) {
            double value;
            if (sample((x+0.5)/24, (y+0.5)/32, value)) {
                dark += value <= dark_threshold;
                ++total;
            }
        }
    if (total < 700) return {};
    const double dark_fraction = static_cast<double>(dark)/total;
    if (dark_fraction < 0.55) return {};

    int border_dark = 0, border_total = 0, good_sides = 0;
    double contrast_sum = 0;
    double edge_contrast_sum = 0;
    int contrast_count = 0;
    for (int side = 0; side < 4; ++side) {
        std::vector<double> differences;
        std::vector<double> edge_differences;
        for (int k = 0; k < 24; ++k) {
            const double t = 0.08 + 0.84 * k/23;
            double inside = 0, outside = 0;
            bool in_ok, out_ok;
            if (side == 0) { in_ok=sample(t,0.025,inside); out_ok=sample(t,-0.025,outside); }
            else if (side == 1) { in_ok=sample(0.975,t,inside); out_ok=sample(1.025,t,outside); }
            else if (side == 2) { in_ok=sample(t,0.975,inside); out_ok=sample(t,1.025,outside); }
            else { in_ok=sample(0.025,t,inside); out_ok=sample(-0.025,t,outside); }
            if (in_ok) { border_dark += inside <= dark_threshold; ++border_total; }
            if (in_ok && out_ok) differences.push_back(outside-inside);
            // 靠近候选边线再采一组点，避免把反光导致的内部阈值线当成纸边。
            if (side == 0) { in_ok=sample(t,0.006,inside); out_ok=sample(t,-0.006,outside); }
            else if (side == 1) { in_ok=sample(0.994,t,inside); out_ok=sample(1.006,t,outside); }
            else if (side == 2) { in_ok=sample(t,0.994,inside); out_ok=sample(t,1.006,outside); }
            else { in_ok=sample(0.006,t,inside); out_ok=sample(-0.006,t,outside); }
            if (in_ok && out_ok) edge_differences.push_back(outside-inside);
        }
        if (differences.size() >= 12) {
            std::sort(differences.begin(), differences.end());
            const double median = differences[differences.size()/2];
            good_sides += median >= 8;
            contrast_sum += median;
            ++contrast_count;
        }
        if (edge_differences.size() >= 12) {
            std::sort(edge_differences.begin(), edge_differences.end());
            edge_contrast_sum += edge_differences[edge_differences.size()/2];
        }
    }
    if (border_total < 80 || static_cast<double>(border_dark)/border_total < 0.65 ||
        good_sides < 2 || contrast_count < 3 || contrast_sum/contrast_count < 8) return {};
    const double shape_score = std::exp(-3*std::abs(std::log(aspect/(210.0/297.0))));
    const double contrast_score = std::clamp(contrast_sum/contrast_count/80, 0.0, 1.0);
    const double edge_score = std::clamp(edge_contrast_sum/4/std::max(15.0,dark_threshold*0.5),0.0,1.0);
    return {0.20*dark_fraction + 0.35*std::sqrt(area_ratio) + 0.15*contrast_score +
            0.20*edge_score + 0.10*shape_score, true};
}
}

A4Detection detect_black_a4(const Mat& image) {
    A4Detection result;
    if (image.empty() || image.type() != CV_8UC3 || std::min(image.cols,image.rows) < 100) return result;
    result.debug_image = image.clone();
    Mat gray, blurred;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(5,5), 0);
    const double otsu = threshold(blurred, result.dark_mask, 0, 255, THRESH_BINARY_INV|THRESH_OTSU);
    // 跟随曝光，同时保留低对比度边缘，补上少量断线。
    const double high = std::clamp(1.5*otsu, 30.0, 180.0);
    Canny(blurred, result.edges, high*0.4, high);
    Mat strong_edges;
    Canny(blurred, strong_edges, 100, 180);
    bitwise_or(result.edges, strong_edges, result.edges);
    const auto kernel = getStructuringElement(MORPH_RECT, Size(5,5));
    Mat closed_edges;
    morphologyEx(result.edges, closed_edges, MORPH_CLOSE, kernel);
    std::vector<Mat> masks = {closed_edges};
    for (double factor : {0.75,1.0,1.20}) {
        Mat mask;
        threshold(blurred, mask, std::clamp(otsu*factor, 15.0, 160.0), 255, THRESH_BINARY_INV);
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);
        masks.push_back(mask);
    }
    const double dark_threshold = std::clamp(otsu+15.0, 45.0, 170.0);
    for (const auto& mask : masks) {
        std::vector<std::vector<Point>> contours;
        // LIST 保留嵌套轮廓，防止导轨/电线的大轮廓把纸张排除。
        findContours(mask, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
        for (const auto& contour : contours) {
            if (std::abs(contourArea(contour)) < image.total()*0.06) continue;
            for (double epsilon : {0.012,0.02,0.035}) {
                std::vector<Point> approx;
                approxPolyDP(contour, approx, arcLength(contour,true)*epsilon, true);
                if (approx.size() != 4) continue;
                std::vector<Point2f> corners(approx.begin(), approx.end());
                corners = orderPoints(corners);
                const auto evidence = evaluate(blurred,corners,dark_threshold);
                if (!evidence.accepted) continue;
                if (evidence.score > result.score) {
                    result.score = evidence.score;
                    result.corners = corners;
                }
            }
        }
    }
    if (result.corners.empty()) return result;
    // 确保纸张短边映射为 210 mm，避免横放时强行拉伸。
    auto& c = result.corners;
    if (norm(c[1]-c[0])+norm(c[3]-c[2]) > norm(c[2]-c[1])+norm(c[0]-c[3]))
        std::rotate(c.begin(),c.begin()+1,c.end());
    result.found = true;
    for (int i = 0; i < 4; ++i) {
        line(result.debug_image,c[i],c[(i+1)%4],Scalar(0,255,0),3,LINE_AA);
        circle(result.debug_image,c[i],5,Scalar(0,0,255),FILLED,LINE_AA);
        putText(result.debug_image,std::to_string(i),c[i]+Point2f(8,20),FONT_HERSHEY_SIMPLEX,0.7,Scalar(0,0,255),2);
    }
    return result;
}

Mat a4_perspective_transform(const Mat& image) {
    const auto detection = detect_black_a4(image);
    if (!detection.edges.empty()) imwrite("edged_origin.png", detection.edges);
    if (!detection.dark_mask.empty()) imwrite("a4_dark_mask.png", detection.dark_mask);
    if (!detection.debug_image.empty()) imwrite("a4_detection.png", detection.debug_image);
    if (!detection.found) {
        std::cerr << "未检测到可靠的黑色 A4 纸边界，请查看 a4_detection.png 和 a4_dark_mask.png" << std::endl;
        return {};
    }
    const std::vector<Point2f> destination = {{0,0},{papers_width*scale,0},
        {papers_width*scale,papers_height*scale},{0,papers_height*scale}};
    const Mat map = getPerspectiveTransform(detection.corners,destination);
    Mat warped;
    warpPerspective(image,warped,map,Size(papers_width*scale,papers_height*scale));
    // 沿用原来的 5 像素裁边和输出尺寸，不改变后续拼块坐标约定。
    const int margin = 5;
    warped = warped(Rect(margin,margin,warped.cols-2*margin,warped.rows-2*margin)).clone();
    imwrite("a4_warped.jpg",warped);
    return warped;
}
