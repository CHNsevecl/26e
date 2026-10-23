#pragma once

#include "puzzle_solver.hpp"
#include <string>

// 单位均为像素；独立保存机械摆放位置，不改变原始拼图解。
struct MachineResult {
    bool ready = false;
    double gap_pixels = 0.0;
    double minimum_gap_pixels = 0.0; // 少于两块时为 0。
    cv::Rect2f bounds;
    std::vector<PuzzlePlacement> placements;
    std::string error;
};

// 保持形状、旋转和源编号，仅将重心向外展开；必要时整体平移以放入 workspace。
// gap_pixels 是任意两块边缘之间的最小距离，不是重心距离。
MachineResult make_machine_result(const PuzzleResult& result, double gap_pixels,
                                 const cv::Rect2f& workspace);
