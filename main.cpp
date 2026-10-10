#include "perspective.hpp"
#include <opencv4/opencv2/opencv.hpp>

int main() {
    setenv("DISPLAY", ":0", 1); // 设置 DISPLAY 环境变量，确保 GUI 能正常显示
    // std::string pipeline = 
    //     "libcamerasrc camera-name=/base/axi/pcie@1000120000/rp1/i2c@88000/imx708@1a ! "
    //     "video/x-raw, format=NV12, width=1280, height=1080 ! "
    //     "videoconvert ! "
    //     "video/x-raw, format=BGR ! "
    //     "appsink drop=true max-buffers=1";

    // cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    // if (!cap.isOpened()) {
    //     std::cerr << "无法打开摄像头" << std::endl;
    //     return -1;
    // }
    // for (int i = 0; i < 30; ++i) {
    //     cv::Mat frame;
    //     cap >> frame; // 丢弃前30帧
    // }
    // 1. 读取图像
    cv::Mat img;
    img = cv::imread("/home/sevecl/Desktop/C/Project/26e/build/original.jpg", cv::IMREAD_COLOR);
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

    task(warped);
    return 0;
}