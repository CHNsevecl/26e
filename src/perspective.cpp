#include "perspective.hpp"
#include "piece_process.hpp"
#include "puzzle_solver.hpp"
#include "machine_result.hpp"
#include "recognition_result.hpp"
#include "card_puzzle.hpp"

#include <iomanip>
#include <sstream>

using namespace cv;
using namespace std;

bool task(const cv::Mat& warped) {
    if (warped.empty() || warped.type() != CV_8UC3) {
        std::cerr << "拼图失败：输入必须是非空 BGR 图像" << std::endl;
        return false;
    }

    cv::Mat gray, blurred, edged, binary;
    std::vector<std::vector<line_data>> lines;
    cv::Mat recognition = cv::Mat::zeros(warped.size(), CV_8UC1);
    const piece_data pieces = line_process(warped, gray, blurred, edged, binary, recognition, lines);

    piece_data pieces_origin = pieces; // 复制一份用于后续处理
    gravity_calculation(pieces_origin, recognition); // 填入拼接前的原始重心，按 source_index 对应最终结果。

    // 当前测试范围：宽 70–120 mm、高 50–90 mm，保持原来的显示位置和像素比例。
    // 第二题正式范围的宽度下限为 90 mm；正式测评时将下面的 70.0f 改回 90.0f。
    const cv::Point2f target_origin(240, 240);
    const cv::Size2f min_size(70.0f * scale, 50.0f * scale);
    const cv::Size2f max_size(120.0f * scale, 90.0f * scale);
    const PuzzleOptions options;
    const auto solve_start = cv::getTickCount();
    const PuzzleResult result = solve_puzzle_in_range(pieces.pieces_point, target_origin, min_size, max_size, options);
    const double solve_time_ms = (cv::getTickCount() - solve_start) * 1000.0 / cv::getTickFrequency();
    const double machine_gap_mm = 5.0; // 机械摆放时，拼块边缘之间至少留 1 cm。
    const MachineResult result_machine = make_machine_result(
        result, machine_gap_mm * scale,
        cv::Rect2f(0, 0, papers_width * scale, papers_height * scale));
    const cv::Rect2f target = result.target;
    double input_area = 0.0;
    int input_count = 0;
    for (const auto& polygon : pieces.pieces_point) {
        if (polygon.empty()) continue;
        ++input_count;
        input_area += std::abs(cv::contourArea(polygon));
    }

    // 最终画布仅包含确认成功的结果，不累积原位置和被否决的候选。
    cv::Mat final_image = cv::Mat::zeros(cv::Size(papers_width * scale, papers_height * scale), CV_8UC3);
    if (result.solved)
        cv::rectangle(final_image, cv::Point2f(target.x, target.y),
                      cv::Point2f(target.x + target.width, target.y + target.height), cv::Scalar(120, 120, 120), 1);
    const std::vector<cv::Scalar> colors = {
        {80, 220, 255}, {100, 255, 100}, {255, 180, 100}, {220, 100, 255}
    };

    cv::FileStorage report("puzzle_result.yml", cv::FileStorage::WRITE);
    if (report.isOpened()) {
        report << "solved" << static_cast<int>(result.solved)
               << "search_limit_reached" << static_cast<int>(result.search_limit_reached)
               << "explored_nodes" << static_cast<double>(result.explored_nodes)
               << "solve_time_ms" << solve_time_ms
               << "input_piece_count" << input_count
               << "input_area_pixels" << input_area
               << "coverage" << result.coverage
               << "overlap_area_pixels" << result.overlap_area
               << "outside_area_pixels" << result.outside_area
               << "position_tolerance_pixels" << options.position_tolerance
               << "target" << target
               << "target_size_mm" << cv::Size2f(target.width / scale, target.height / scale)
               << "target_min_size_mm" << cv::Size2f(min_size.width / scale, min_size.height / scale)
               << "target_max_size_mm" << cv::Size2f(max_size.width / scale, max_size.height / scale)
               << "min_coverage" << options.min_coverage
               << "max_overlap_ratio" << options.max_overlap_ratio
               << "max_outside_ratio" << options.max_outside_ratio
               << "source_polygons" << "[";
        for (std::size_t i = 0; i < pieces.pieces_point.size(); ++i)
            if (!pieces.pieces_point[i].empty())
                report << "{" << "source_index" << static_cast<int>(i) << "points" << pieces.pieces_point[i] << "}";
        report << "]" << "placements" << "[";
    }
    for (const auto& placement : result.placements) {
        const auto color = colors[placement.source_index % colors.size()];
        for (std::size_t j = 0; j < placement.points.size(); ++j) {
            cv::line(final_image, placement.points[j], placement.points[(j + 1) % placement.points.size()], color, 1, cv::LINE_AA);
            cv::circle(final_image, placement.points[j], 2, color, cv::FILLED, cv::LINE_AA);
        }
        cv::circle(final_image, placement.center_gravity, 5, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
        if (report.isOpened()) {
            report << "{" << "source_index" << static_cast<int>(placement.source_index)
                   << "rotation_degrees" << placement.rotation_radians * 180.0 / CV_PI
                   << "center_gravity" << placement.center_gravity
                   << "translation" << placement.translation << "points" << placement.points << "}";
        }
    }
    if (report.isOpened()) {
        report << "]" << "result_machine" << "{"
               << "ready" << static_cast<int>(result_machine.ready)
               << "gap_mm" << machine_gap_mm
               << "minimum_gap_mm" << result_machine.minimum_gap_pixels / scale
               << "bounds" << result_machine.bounds
               << "error" << result_machine.error << "placements" << "[";
    }
    cv::Mat machine_image = cv::Mat::zeros(final_image.size(), CV_8UC3);
    for (const auto& placement : result_machine.placements) {
        const auto color = colors[placement.source_index % colors.size()];
        for (std::size_t j = 0; j < placement.points.size(); ++j)
            cv::line(machine_image, placement.points[j], placement.points[(j + 1) % placement.points.size()],
                     color, 1, cv::LINE_AA);
        cv::circle(machine_image, placement.center_gravity, 5, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
        if (report.isOpened())
            report << "{" << "source_index" << static_cast<int>(placement.source_index)
                   << "rotation_degrees" << placement.rotation_radians * 180.0 / CV_PI
                   << "center_gravity" << placement.center_gravity
                   << "translation" << placement.translation << "points" << placement.points << "}";
    }
    if (report.isOpened()) {
        report << "]" << "}";
        report.release();
    }
    if (!cv::imwrite("lines.png", final_image)) std::cerr << "无法保存 lines.png" << std::endl;
    if (!cv::imwrite("lines_machine.png", machine_image)) std::cerr << "无法保存 lines_machine.png" << std::endl;

    if (result.solved) {
        std::cout << "拼图完成：" << result.placements.size() << " 块，矩形尺寸 "
                  << target.width / scale << " × " << target.height / scale << " mm，覆盖率 "
                  << result.coverage * 100.0 << "% ，重叠面积 " << result.overlap_area
                  << " 像素²，框外面积 " << result.outside_area << " 像素²" << std::endl;
    } else {
        std::cerr << (result.search_limit_reached ? "搜索达到上限，尚未找到满足容差的拼法" : "未找到满足覆盖、重叠和边界约束的拼法")
                  << "；请检查识别轮廓和 puzzle_result.yml" << std::endl;
    }

    if (!result_machine.ready) {
        std::cerr << "未发送机械指令：" << result_machine.error << std::endl;
        return false;
    }
    std::cout << "机械布局完成，拼块最小间距 " << result_machine.minimum_gap_pixels / scale << " mm" << std::endl;
    return save_machine_targets(pieces_origin, result_machine, "task1");
}

bool task2(const cv::Mat& warped) {
    CardPuzzleOptions options;
    options.gap_mm = 5.0; // 扑克牌机械目标的边缘最小间距，单位 mm。
    // 已知完整牌尺寸时填写，例如 options.target_size_mm = {63.0f, 88.0f};
    // 保持为零则根据碎片面积、边长和 min_aspect/max_aspect 估计。
    const CardPuzzleResult card = solve_card_puzzle(warped, options);
    const piece_data& pieces_origin = card.recognition.pieces;
    const PuzzleResult& result = card.result;
    const MachineResult& result_machine = card.result_machine;
    if (!card.recognition.foreground_mask.empty()) cv::imwrite("task2_mask.png", card.recognition.foreground_mask);
    if (!card.recognition.background_image.empty()) cv::imwrite("task2_background.png", card.recognition.background_image);
    if (!card.recognition.debug_image.empty()) cv::imwrite("task2_recognition.png", card.recognition.debug_image);
    const cv::Mat empty_canvas = cv::Mat::zeros(papers_height * scale, papers_width * scale, CV_8UC3);
    cv::imwrite("task2_result.png", card.assembled_image.empty() ? empty_canvas : card.assembled_image);
    cv::imwrite("task2_machine.png", card.machine_image.empty() ? empty_canvas : card.machine_image);

    cv::FileStorage report("task2_result.yml", cv::FileStorage::WRITE);
    auto save_placements = [&](const std::vector<PuzzlePlacement>& placements) {
        report << "[";
        for (const auto& p : placements)
            report << "{" << "source_index" << static_cast<int>(p.source_index)
                   << "rotation_radians" << p.rotation_radians
                   << "rotation_degrees" << p.rotation_radians * 180.0 / CV_PI
                   << "translation" << p.translation << "center_gravity" << p.center_gravity
                   << "points" << p.points << "}";
        report << "]";
    };
    if (report.isOpened()) {
        report << "recognized" << static_cast<int>(card.recognition.ready)
               << "solved" << static_cast<int>(result.solved)
               << "error" << card.error
               << "matching_method" << "geometry_and_seam_continuity_without_reference"
               << "contour_model" << "fitted_edges_with_physical_rounded_mask"
               << "unique_solution_verified" << 0
               << "input_piece_count" << static_cast<int>(pieces_origin.pieces_point.size())
               << "search_limit_reached" << static_cast<int>(result.search_limit_reached)
               << "explored_nodes" << static_cast<double>(result.explored_nodes)
               << "evaluated_solutions" << static_cast<double>(result.evaluated_solutions)
               << "seam_score" << result.seam_score
               << "coverage" << result.coverage
               << "overlap_area_pixels" << result.overlap_area
               << "outside_area_pixels" << result.outside_area
               << "target" << result.target
               << "target_size_mm" << cv::Size2f(result.target.width/scale,result.target.height/scale)
               << "gap_mm" << options.gap_mm << "source_polygons" << "[";
    }
    for (const auto& fragment : card.recognition.fragments) {
        const auto i = fragment.source_index;
        const std::string filename = "task2_piece_" + std::to_string(i);
        cv::imwrite(filename + ".png", fragment.image);
        cv::imwrite(filename + "_mask.png", fragment.mask);
        if (report.isOpened())
            report << "{" << "source_index" << static_cast<int>(i)
                   << "points" << pieces_origin.pieces_point[i]
                   << "center_gravity" << pieces_origin.center_gravity[i]
                   << "bounds" << fragment.bounds
                   << "image" << filename + ".png" << "mask" << filename + "_mask.png" << "}";
    }
    if (report.isOpened()) {
        report << "]" << "placements";
        save_placements(result.placements);
        report << "result_machine" << "{" << "ready" << static_cast<int>(result_machine.ready)
               << "minimum_gap_mm" << result_machine.minimum_gap_pixels/scale
               << "bounds" << result_machine.bounds << "placements";
        save_placements(result_machine.placements);
        report << "}";
        report.release();
    }
    if (!result.solved || !result_machine.ready) {
        std::cerr << "task2：" << card.error << std::endl;
        return false;
    }
    std::cout << "扑克牌候选复原完成：" << result.placements.size() << " 块，完整牌尺寸 "
              << result.target.width/scale << " × " << result.target.height/scale
              << " mm，接缝图案评分 " << result.seam_score
              << "，机械最小间距 " << result_machine.minimum_gap_pixels/scale << " mm" << std::endl;
    std::cout << "无参考图的接缝匹配不保证唯一解，牌面预览保存在 task2_result.png" << std::endl;
    return save_machine_targets(pieces_origin, result_machine, "task2");
}
