#include "perspective.hpp"
#include "recognition_result.hpp"
#include <filesystem>

int main(int argc, char** argv) {
    try {
        bool card_mode = false;
        std::string image_path, output_dir, send_result, serial_device = "/dev/ttyAMA0";
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--task1") card_mode = false;
            else if (argument == "--task2") card_mode = true;
            else if (argument == "--dry-run") {} // Compatibility: recognition is always dry.
            else if (argument == "--image" && i + 1 < argc) image_path = argv[++i];
            else if (argument == "--output-dir" && i + 1 < argc) output_dir = argv[++i];
            else if (argument == "--send-result" && i + 1 < argc) send_result = argv[++i];
            else if (argument == "--serial-device" && i + 1 < argc) serial_device = argv[++i];
            else {
                std::cerr << "用法：ps [--task1|--task2] [--image 图片] [--output-dir 目录]\n"
                             "      ps --send-result machine_targets.json [--serial-device /dev/ttyAMA0]\n";
                return argument == "--help" ? 0 : 2;
            }
        }
        // This branch never captures a new photo or re-runs recognition.
        if (!send_result.empty()) return send_saved_targets(send_result, serial_device) ? 0 : 2;
        if (!image_path.empty()) image_path = std::filesystem::absolute(image_path).string();
        if (!output_dir.empty()) {
            std::filesystem::create_directories(output_dir);
            std::filesystem::current_path(output_dir);
        }
        // A failed attempt must never expose an earlier batch as its own result.
        std::filesystem::remove("machine_targets.json");
        cv::Mat img;
        if (!image_path.empty()) img = cv::imread(image_path, cv::IMREAD_COLOR);
        else {
            const std::string pipeline =
                "libcamerasrc camera-name=/base/axi/pcie@1000120000/rp1/i2c@88000/imx708@1a ! "
                "video/x-raw, format=NV12, width=1280, height=1080 ! "
                "videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1";
            cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
            if (!cap.isOpened()) {
                std::cerr << "无法打开摄像头" << std::endl;
                return 1;
            }
            for (int i = 0; i < 30; ++i) cap >> img;
            cap >> img;
        }
        if (img.empty()) {
            std::cerr << "无法读取图像，请检查路径或摄像头" << std::endl;
            return 1;
        }
        cv::imwrite("original.jpg", img);
        const cv::Mat warped = a4_perspective_transform(img);
        if (warped.empty()) {
            std::cerr << "透视变换失败" << std::endl;
            return 1;
        }
        const bool ready = card_mode ? task2(warped) : task(warped);
        if (ready) std::cout << "识别成功，机械目标已保存，等待手动发送。" << std::endl;
        return ready ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "处理异常：" << error.what() << std::endl;
        return 2;
    }
}
