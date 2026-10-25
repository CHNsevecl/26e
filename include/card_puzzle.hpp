#pragma once

#include "perspective.hpp"
#include "puzzle_solver.hpp"
#include "machine_result.hpp"

struct CardPuzzleOptions {
    double min_piece_area_mm2 = 30.0;
    double closing_mm = 1.0; // 小范围连接印刷笔画造成的断裂，不能超过牌片实际间距。
    cv::Size2f target_size_mm; // 已知完整牌尺寸时填写；为零时按面积、边长估计。
    double min_aspect = 0.50; // 未知尺寸时，短边/长边的候选范围。
    double max_aspect = 0.85;
    double gap_mm = 5.0;
    PuzzleOptions solver;
    bool use_white_border = true; // 普通白边扑克牌；无白边牌面可关闭此外观先验。
};

struct CardFragment {
    std::size_t source_index = 0;
    cv::Rect bounds;
    cv::Mat image; // 保留原始牌面颜色，坐标相对 bounds。
    cv::Mat mask;  // 包括内部黑字/黑色花纹，不把这些区域当作孔洞。
};

struct CardRecognition {
    bool ready = false;
    piece_data pieces; // points 是直边几何模型，center_gravity 是实体轮廓重心。
    cv::Mat foreground_mask;
    cv::Mat background_image; // 黑纸在不同位置的亮度/颜色估计。
    cv::Mat debug_image;
    std::vector<CardFragment> fragments;
    std::string error;
};

struct CardPuzzleResult {
    CardRecognition recognition;
    PuzzleResult result;
    MachineResult result_machine;
    cv::Mat assembled_image;
    cv::Mat machine_image;
    std::string error;
};

CardRecognition recognize_card_pieces(const cv::Mat& warped, const CardPuzzleOptions& options = {});
double card_seam_score(const CardRecognition& recognition, const PuzzlePlacement& a,
                       const PuzzlePlacement& b, double tolerance_pixels = 5.0);
CardPuzzleResult solve_card_puzzle(const cv::Mat& warped, const CardPuzzleOptions& options = {});
