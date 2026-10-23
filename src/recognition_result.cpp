#include "recognition_result.hpp"
#include "uart.hpp"

#include <cmath>
#include <iomanip>
#include <set>
#include <sstream>

namespace {
bool valid_point(const cv::Point2f& point) {
    return std::isfinite(point.x) && std::isfinite(point.y) &&
           point.x >= 0 && point.y >= 0 &&
           point.x <= papers_width * scale && point.y <= papers_height * scale;
}
}

bool save_machine_targets(const piece_data& origin, const MachineResult& result,
                          const std::string& task_name, const std::string& path) {
    if (!result.ready || result.placements.empty()) return false;
    std::set<std::size_t> ids;
    for (const auto& p : result.placements)
        if (p.source_index >= origin.center_gravity.size() || !ids.insert(p.source_index).second ||
            !valid_point(origin.center_gravity[p.source_index]) || !valid_point(p.center_gravity) ||
            !std::isfinite(p.rotation_radians)) return false;
    cv::FileStorage report(path, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    if (!report.isOpened()) return false;
    report << "version" << 1 << "ready" << 1 << "task" << task_name << "targets" << "[";
    for (const auto& p : result.placements)
        report << "{" << "source_index" << static_cast<int>(p.source_index)
               << "origin_pixels" << origin.center_gravity[p.source_index]
               << "target_pixels" << p.center_gravity
               << "rotation_degrees" << p.rotation_radians * 180.0 / CV_PI << "}";
    report << "]";
    report.release();
    return true;
}

bool send_saved_targets(const std::string& path, const std::string& device) {
    cv::FileStorage report(path, cv::FileStorage::READ);
    if (!report.isOpened() || static_cast<int>(report["version"]) != 1 ||
        static_cast<int>(report["ready"]) != 1) return false;
    const auto targets = report["targets"];
    if (!targets.isSeq() || targets.empty()) return false;
    std::set<int> ids;
    std::ostringstream commands;
    commands << std::fixed << std::setprecision(1);
    // Validate the entire batch before opening UART or sending its first byte.
    for (const auto& item : targets) {
        const auto id_node = item["source_index"];
        const auto rotation_node = item["rotation_degrees"];
        const auto from = item["origin_pixels"], to = item["target_pixels"];
        if (!id_node.isInt() || (!rotation_node.isInt() && !rotation_node.isReal()) ||
            !from.isSeq() || from.size() != 2 || !to.isSeq() || to.size() != 2) return false;
        for (const auto& p : {from, to})
            for (const auto& value : p)
                if (!value.isInt() && !value.isReal()) return false;
        const int id = static_cast<int>(id_node);
        const double angle = static_cast<double>(rotation_node);
        const cv::Point2f origin(static_cast<float>(from[0]), static_cast<float>(from[1]));
        const cv::Point2f target(static_cast<float>(to[0]), static_cast<float>(to[1]));
        if (id < 0 || !ids.insert(id).second || !std::isfinite(angle) ||
            !valid_point(origin) || !valid_point(target)) return false;
        commands << id << " 0 x:" << origin.y / scale << " y:" << origin.x / scale << " r:" << angle << ";\n"
                 << id << " 1 x:" << target.y / scale << " y:" << target.x / scale << ";\n";
    }
    commands << "finish\n";
    UART uart(device, 9600);
    if (!uart.open() || !uart.send(commands.str())) return false;
    std::cout << "本次机械目标已发送。" << std::endl;
    return true;
}
