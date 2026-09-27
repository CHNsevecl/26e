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
        if (area < 1000) continue; // 过滤小轮廓

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
        Point2f(papers_width, 0),
        Point2f(papers_width, papers_height),
        Point2f(0, papers_height)
    };

    // 6. 计算透视变换矩阵
    Mat M = getPerspectiveTransform(srcPts, dstPts);

    // 7. 执行透视变换，输出尺寸 210×297（1像素 = 1毫米）
    Mat warped;
    warpPerspective(img, warped, M, Size(papers_width, papers_height));

    // 9. 保存结果
    imwrite("a4_warped.jpg", warped);

    return warped;
}
