#pragma once

#include <opencv4/opencv2/opencv.hpp>
#include <cstddef>
#include <functional>
#include <vector>

struct PuzzlePlacement;

// 所有长度和面积均使用透视校正后的像素坐标。
struct PuzzleOptions {
    double position_tolerance = 5.0;
    double angle_tolerance_degrees = 4.0;
    double min_coverage = 0.97;
    double max_overlap_ratio = 0.02;
    double max_outside_ratio = 0.01;
    std::size_t max_nodes = 30000;
    double time_limit_seconds = 8.0;
    // 可选的接缝图案评分，越高越好。为空时保持原有几何求解行为。
    std::function<double(const PuzzlePlacement&, const PuzzlePlacement&)> seam_score;
};

struct PuzzlePlacement {
    std::size_t source_index = 0;
    // x' = cos(a)*x - sin(a)*y，y' = sin(a)*x + cos(a)*y；图像坐标中正角为顺时针。
    double rotation_radians = 0.0;
    cv::Point2f translation;
    std::vector<cv::Point2f> points;
    cv::Point2f center_gravity; // 最终多边形的面积重心，单位：像素。
};

struct PuzzleResult {
    bool solved = false;
    bool search_limit_reached = false;
    std::size_t explored_nodes = 0;
    double coverage = 0.0;
    double overlap_area = 0.0;
    double outside_area = 0.0;
    std::vector<PuzzlePlacement> placements;
    cv::Rect2f target; // 实际采用的矩形，像素坐标；自动尺寸求解失败时为空。
    double seam_score = 0.0;
    std::size_t evaluated_solutions = 0; // 图案模式下经过几何验收的候选数，不代表唯一解。
};

// 输入保持不变；每次候选独立验证，失败后回退。只旋转、平移，不缩放或翻面。
PuzzleResult solve_puzzle(const std::vector<std::vector<cv::Point2f>>& polygons,
                          const cv::Rect2f& target,
                          const PuzzleOptions& options = PuzzleOptions{});

// 在宽高范围内同时搜索尺寸和拼法，所有尺寸共用同一份时间/节点预算。
PuzzleResult solve_puzzle_in_range(const std::vector<std::vector<cv::Point2f>>& polygons,
                                   cv::Point2f origin, cv::Size2f min_size, cv::Size2f max_size,
                                   const PuzzleOptions& options = PuzzleOptions{});
