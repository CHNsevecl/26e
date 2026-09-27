#include "perspective.hpp"
#include <opencv4/opencv2/opencv.hpp>

int main() {
    setenv("DISPLAY", ":0", 1); // 设置 DISPLAY 环境变量，确保 GUI 能正常显示
    std::string pipeline = 
        "libcamerasrc camera-name=/base/axi/pcie@1000120000/rp1/i2c@88000/imx708@1a ! "
        "video/x-raw, format=NV12, width=1920, height=1080 ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=true max-buffers=1";

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头" << std::endl;
        return -1;
    }
    for (int i = 0; i < 30; ++i) {
        cv::Mat frame;
        cap >> frame; // 丢弃前30帧
    }
    // 1. 读取图像
    cv::Mat img;
    cap >> img;
    if (img.empty()) {
        std::cout << "无法读取图像，请检查路径" << std::endl;
        return -1;
    }

    cv::imwrite("original.jpg", img);

    cv::Mat warped = a4_perspective_transform(img);

    if (warped.empty()) {
        std::cout << "透视变换失败" << std::endl;
        return -1;
    }

    cv::Mat gray, blurred, edged;
    cv::cvtColor(warped, gray, cv::COLOR_BGR2GRAY);
    // 假设原始设计是 210×297 下用 kernel=5, canny=50/150
    double scale = warped.cols / 210.0;  // 2100/210 = 10

    int k = cvRound(5 * scale);          // 50
    if (k % 2 == 0) k += 1;              // 保证奇数
    int low  = cvRound(50 * scale);      // 500
    int high = cvRound(150 * scale);     // 1500

    // cv::GaussianBlur(gray, blurred, cv::Size(k, k), 0);
    // cv::imwrite("debug_blurred.png", blurred);
    cv::imwrite("debug_gray.png", gray);
    cv::Canny(gray, edged, 50, 150);

    cv::imwrite("debug_edged.png", edged);
    cv::imwrite("debug_warped.png", warped);    

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edged, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point>> approxContours;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1));

    for (auto& c : contours) {
        std::vector<cv::Point> approx;
        double peri = cv::arcLength(c, true);
        cv::approxPolyDP(c, approx, 0.01 * peri, true);

        // 把简化后的轮廓画到临时黑图上，做膨胀，再提取回来
        cv::Mat tmp = cv::Mat::zeros(edged.size(), CV_8UC1);
        std::vector<std::vector<cv::Point>> tmpContours = { approx };
        cv::drawContours(tmp, tmpContours, -1, cv::Scalar(255), 1); // 线宽 1
        cv::dilate(tmp, tmp, kernel);

        std::vector<std::vector<cv::Point>> dilatedContours;
        cv::findContours(tmp, dilatedContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (!dilatedContours.empty()) {
            approxContours.push_back(dilatedContours[0]);
        }
    }

    cv::Mat contourImg = cv::Mat::zeros(edged.size(), CV_8UC1);

    // 画所有简化后的轮廓，白色，线宽 1
    cv::drawContours(contourImg, approxContours, -1, cv::Scalar(255), 1);

    cv::imshow("Warped A4", warped); // 透视变换后的
    cv::imshow("Original Edges", edged);        // 原始 Canny 边缘（锯齿）
    cv::imshow("Smoothed Contours", contourImg); // 简化后的轮廓
    cv::waitKey(0);


    cv::waitKey(0);
}