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
    bool use_rounded_corners = true; // 从实体掩膜检测原牌圆角，作为外框位置的软证据。
    bool use_layout_pattern = true; // Compare paired corner indices and the whole printed face.
    bool require_corner_indices = false; // task2 requires a verified pair before enabling manual send.
};

struct CardFragment {
    std::size_t source_index = 0;
    cv::Rect bounds;
    cv::Mat image; // 保留原始牌面颜色，坐标相对 bounds。
    cv::Mat mask;  // 包括内部黑字/黑色花纹，不把这些区域当作孔洞。
    cv::Mat normalized_ink; // Per-fragment paper-white normalization for whole-face comparison.
    cv::Mat texture_mask; // Fitted cut texture may include ink omitted by the physical mask.
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

struct CardLayoutEvidence {
    bool first_index = false;
    bool opposite_index = false;
    bool indices_verified = false;
    double index_match = 0;
    double pattern_match = 0;
    double score = 0;
    double rank_match = 0;
    double suit_match = 0;
    double index_aspect_ratio = 0;
};

struct CardPuzzleResult {
    CardRecognition recognition;
    PuzzleResult result;
    MachineResult result_machine;
    cv::Mat assembled_image;
    cv::Mat machine_image;
    std::string error;
    CardLayoutEvidence layout;
};

CardRecognition recognize_card_pieces(const cv::Mat& warped, const CardPuzzleOptions& options = {});
double card_seam_score(const CardRecognition& recognition, const PuzzlePlacement& a,
                       const PuzzlePlacement& b, double tolerance_pixels = 5.0);
CardPuzzleResult solve_card_puzzle(const cv::Mat& warped, const CardPuzzleOptions& options = {});

CardLayoutEvidence evaluate_card_layout(const CardRecognition& recognition,
    const std::vector<PuzzlePlacement>& placements, const cv::Rect2f& target);
