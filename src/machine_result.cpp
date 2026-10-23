#include "machine_result.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
double cross(cv::Point2d a, cv::Point2d b) { return a.x * b.y - a.y * b.x; }

double point_segment(cv::Point2d p, cv::Point2d a, cv::Point2d b) {
    const auto v = b - a;
    const double length2 = v.dot(v);
    const double t = length2 > 0 ? std::clamp((p - a).dot(v) / length2, 0.0, 1.0) : 0.0;
    return cv::norm(p - (a + t * v));
}

double polygon_distance(const std::vector<cv::Point2f>& a, const std::vector<cv::Point2f>& b) {
    if (cv::pointPolygonTest(a, b.front(), false) >= 0 ||
        cv::pointPolygonTest(b, a.front(), false) >= 0) return 0;
    double distance = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < a.size(); ++i) {
        const cv::Point2d p = a[i], p2 = a[(i + 1) % a.size()];
        for (std::size_t j = 0; j < b.size(); ++j) {
            const cv::Point2d q = b[j], q2 = b[(j + 1) % b.size()];
            const auto u = p2 - p, v = q2 - q;
            const double denominator = cross(u, v);
            if (std::abs(denominator) > 1e-12) {
                const double t = cross(q - p, v) / denominator;
                const double s = cross(q - p, u) / denominator;
                if (t >= 0 && t <= 1 && s >= 0 && s <= 1) return 0;
            }
            distance = std::min({distance, point_segment(p, q, q2), point_segment(p2, q, q2),
                                 point_segment(q, p, p2), point_segment(q2, p, p2)});
        }
    }
    return distance;
}

double minimum_gap(const std::vector<PuzzlePlacement>& placements) {
    if (placements.size() < 2) return 0;
    double distance = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < placements.size(); ++i)
        for (std::size_t j = i + 1; j < placements.size(); ++j)
            distance = std::min(distance, polygon_distance(placements[i].points, placements[j].points));
    return distance;
}

cv::Rect2f layout_bounds(const std::vector<PuzzlePlacement>& placements) {
    float left = std::numeric_limits<float>::infinity(), top = left;
    float right = -left, bottom = -left;
    for (const auto& placement : placements)
        for (const auto& p : placement.points) {
            left = std::min(left, p.x); top = std::min(top, p.y);
            right = std::max(right, p.x); bottom = std::max(bottom, p.y);
        }
    return {left, top, right - left, bottom - top};
}

void translate(PuzzlePlacement& placement, cv::Point2f delta) {
    placement.center_gravity += delta;
    placement.translation += delta;
    for (auto& p : placement.points) p += delta;
}
}

MachineResult make_machine_result(const PuzzleResult& result, double gap_pixels,
                                 const cv::Rect2f& workspace) {
    MachineResult machine;
    machine.gap_pixels = gap_pixels;
    if (!result.solved || result.placements.empty()) {
        machine.error = "原始拼图尚未求解成功";
        return machine;
    }
    if (!std::isfinite(gap_pixels) || gap_pixels <= 0 ||
        !std::isfinite(workspace.x) || !std::isfinite(workspace.y) ||
        !std::isfinite(workspace.width) || !std::isfinite(workspace.height) ||
        workspace.width <= 0 || workspace.height <= 0) {
        machine.error = "机械间距或工作区域无效";
        return machine;
    }
    for (const auto& p : result.placements) {
        if (p.points.size() < 3 || !std::isfinite(p.center_gravity.x) ||
            !std::isfinite(p.center_gravity.y) || !std::isfinite(p.rotation_radians) ||
            !std::isfinite(p.translation.x) || !std::isfinite(p.translation.y)) {
            machine.error = "拼块几何数据无效";
            return machine;
        }
        for (const auto& point : p.points)
            if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
                machine.error = "拼块顶点无效";
                return machine;
            }
    }

    const auto original_bounds = layout_bounds(result.placements);
    const cv::Point2f center = original_bounds.tl() +
        cv::Point2f(original_bounds.width / 2, original_bounds.height / 2);
    auto expanded = [&](double factor) {
        auto placements = result.placements;
        for (auto& placement : placements)
            translate(placement, (placement.center_gravity - center) * static_cast<float>(factor - 1));
        return placements;
    };

    // 在每对重心的连线方向上分离投影，得到保证边缘间距的展开上界。
    // 随后用真实多边形边缘距离缩小展开量，兼容斜边和凹多边形。
    const double required_gap = gap_pixels + 0.01; // 浮点坐标余量。
    double upper = 1.0;
    for (std::size_t i = 0; i < result.placements.size(); ++i) {
        const auto& a = result.placements[i];
        for (std::size_t j = i + 1; j < result.placements.size(); ++j) {
            const auto& b = result.placements[j];
            const cv::Point2d delta = b.center_gravity - a.center_gravity;
            const double length = cv::norm(delta);
            if (length < 1e-6) {
                machine.error = "拼块重心重合，无法按重心方向展开";
                return machine;
            }
            const auto direction = delta / length;
            double a_max = -std::numeric_limits<double>::infinity();
            double b_min = std::numeric_limits<double>::infinity();
            for (const auto& p : a.points)
                a_max = std::max(a_max, cv::Point2d(p - a.center_gravity).dot(direction));
            for (const auto& p : b.points)
                b_min = std::min(b_min, cv::Point2d(p - b.center_gravity).dot(direction));
            upper = std::max(upper, (required_gap + a_max - b_min) / length);
        }
    }
    if (result.placements.size() > 1 && minimum_gap(result.placements) < required_gap) {
        double lower = 1.0;
        upper += 1e-5;
        for (int iteration = 0; iteration < 32; ++iteration) {
            const double middle = (lower + upper) / 2;
            if (minimum_gap(expanded(middle)) >= required_gap) upper = middle;
            else lower = middle;
        }
    } else upper = 1.0;

    auto placements = expanded(upper);
    auto bounds = layout_bounds(placements);
    if (bounds.width > workspace.width || bounds.height > workspace.height) {
        machine.error = "增加间距后的拼块超出工作区域";
        return machine;
    }
    // 优先保持原来的中心位置；只有超出 A4 工作区域时才整体挪回。
    const cv::Point2f correction(
        std::clamp(bounds.x, workspace.x, workspace.x + workspace.width - bounds.width) - bounds.x,
        std::clamp(bounds.y, workspace.y, workspace.y + workspace.height - bounds.height) - bounds.y);
    for (auto& placement : placements) translate(placement, correction);
    machine.minimum_gap_pixels = minimum_gap(placements);
    if (placements.size() > 1 && machine.minimum_gap_pixels < gap_pixels) {
        machine.error = "机械布局未通过最小间距验证";
        return machine;
    }
    machine.bounds = layout_bounds(placements);
    machine.placements = std::move(placements);
    machine.ready = true;
    return machine;
}
