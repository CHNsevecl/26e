#include "a4_detection.hpp"
#include "perspective.hpp"

#include <iostream>
#include <stdexcept>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void check_corners(const cv::Mat& image, const std::vector<cv::Point2f>& expected, double tolerance) {
    const auto detected = detect_black_a4(image);
    require(detected.found, "black A4 not detected");
    require(detected.corners.size() == 4 && cv::isContourConvex(detected.corners), "invalid corners");
    for (const auto& p : expected) {
        double distance = 1e10;
        for (const auto& corner : detected.corners) distance = std::min(distance,cv::norm(p-corner));
        if (distance > tolerance) {
            std::cerr << "Expected " << p << ", detected " << detected.corners << '\n';
            throw std::runtime_error("paper boundary inaccurate or wrong object selected");
        }
    }
}
}

int main(int argc,char** argv) {
    try {
        cv::Mat scene(800,1000,CV_8UC3,cv::Scalar(155,165,175));
        // 桌面杂物、比纸张更大的白色四边形和导轨。
        cv::rectangle(scene,{10,10},{990,790},{230,230,230},15);
        cv::rectangle(scene,{20,100},{85,700},{30,35,40},cv::FILLED);
        cv::rectangle(scene,{900,20},{930,760},{50,55,60},cv::FILLED);
        const std::vector<cv::Point> paper={{260,70},{660,90},{800,730},{150,700}};
        cv::fillConvexPoly(scene,paper,{25,25,25});
        cv::rectangle(scene,{290,330},{420,500},{240,240,240},cv::FILLED);
        cv::fillConvexPoly(scene,std::vector<cv::Point>{{460,580},{640,300},{700,610}},{245,245,245});
        const std::vector<cv::Point2f> expected(paper.begin(),paper.end());
        check_corners(scene,expected,8);
        cv::Mat dim;
        scene.convertTo(dim,-1,0.38,5);
        cv::GaussianBlur(dim,dim,{7,7},1.5);
        check_corners(dim,expected,10);
        cv::Mat noisy = scene.clone();
        cv::RNG random(42);
        for (int y=0;y<noisy.rows;++y)
            for (int x=0;x<noisy.cols;++x)
                for (int c=0;c<3;++c)
                    noisy.at<cv::Vec3b>(y,x)[c] = cv::saturate_cast<uchar>(noisy.at<cv::Vec3b>(y,x)[c]+random.gaussian(9));
        check_corners(noisy,expected,10);
        cv::Mat shadow = scene.clone();
        for (int y=0;y<shadow.rows;++y)
            for (int x=0;x<shadow.cols;++x) {
                const double illumination = 0.4 + 0.6*x/shadow.cols;
                auto& pixel = shadow.at<cv::Vec3b>(y,x);
                for (int c=0;c<3;++c) pixel[c] = cv::saturate_cast<uchar>(pixel[c]*illumination);
            }
        check_corners(shadow,expected,10);
        // 一小段边界被背景颜色打断，不能只依赖完全闭合的 Canny 外轮廓。
        cv::Mat interrupted = scene.clone();
        cv::rectangle(interrupted,{430,67},{460,87},{155,165,175},cv::FILLED);
        check_corners(interrupted,expected,10);
        cv::Mat landscape;
        cv::rotate(scene,landscape,cv::ROTATE_90_CLOCKWISE);
        std::vector<cv::Point2f> landscape_expected;
        for (const auto& p : expected) landscape_expected.push_back({scene.rows-1-p.y,p.x});
        check_corners(landscape,landscape_expected,10);
        cv::Mat no_paper(800,1000,CV_8UC3,cv::Scalar(180,180,180));
        cv::rectangle(no_paper,{200,50},{700,750},{245,245,245},cv::FILLED);
        require(!detect_black_a4(no_paper).found,"white rectangle misidentified as black A4");
        require(!detect_black_a4(cv::Mat::zeros(800,1000,CV_8UC3)).found,"uniform black frame accepted");
        require(!detect_black_a4(cv::Mat{}).found,"empty image accepted");
        // 原 x+y / x-y 法在菱形上可能把同一个点重复当作两个角。
        const auto ordered = orderPoints({{100,0},{200,100},{100,200},{0,100}});
        for (int i=0;i<4;++i)
            for (int j=i+1;j<4;++j) require(cv::norm(ordered[i]-ordered[j])>1,"corner ordering duplicated a vertex");
        if (argc>1) {
            const auto photo = cv::imread(argv[1]);
            require(!photo.empty(),"missing photo fixture");
            const std::vector<cv::Point2f> corners={{390,51},{903,58},{1027,818},{254,812}};
            check_corners(photo,corners,12);
            photo.convertTo(dim,-1,0.45,5);
            check_corners(dim,corners,12);
            const auto result = detect_black_a4(photo);
            cv::imwrite("/tmp/26e-a4-detection.png",result.debug_image);
            const auto warped = a4_perspective_transform(photo);
            require(warped.size()==cv::Size(830,1178),"warp output dimensions changed");
        }
        std::cout << "A4 detection tests passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
