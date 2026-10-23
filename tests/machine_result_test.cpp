#include "machine_result.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

PuzzleResult layout(const std::vector<std::vector<cv::Point2f>>& polygons) {
    PuzzleResult result;
    result.solved = true;
    for (std::size_t i = 0; i < polygons.size(); ++i) {
        PuzzlePlacement p;
        p.source_index = i * 2; // 源编号可以不连续。
        p.points = polygons[i];
        const auto m = cv::moments(p.points);
        p.center_gravity = {static_cast<float>(m.m10 / m.m00), static_cast<float>(m.m01 / m.m00)};
        p.rotation_radians = 0.37 * i;
        p.translation = {12.0f * i, 24.0f * i};
        result.placements.push_back(p);
    }
    return result;
}

void verify(const PuzzleResult& original, cv::Rect2f workspace) {
    const auto saved = original.placements;
    const auto machine = make_machine_result(original, 40, workspace); // 4 px/mm：10 mm。
    require(machine.ready, machine.error.c_str());
    require(machine.placements.size() == saved.size(), "missing piece");
    for (std::size_t i = 0; i < saved.size(); ++i) {
        const auto& a = saved[i];
        const auto& b = machine.placements[i];
        require(a.source_index == b.source_index && a.rotation_radians == b.rotation_radians,
                "source index or rotation changed");
        require(a.points == original.placements[i].points && a.center_gravity == original.placements[i].center_gravity,
                "original layout modified");
        const auto delta = b.center_gravity - a.center_gravity;
        require(cv::norm(b.translation - a.translation - delta) < 0.002, "translation inconsistent with centroid");
        const auto moments = cv::moments(b.points);
        require(cv::norm(b.center_gravity - cv::Point2f(moments.m10 / moments.m00, moments.m01 / moments.m00)) < 0.002,
                "centroid does not match polygon");
        for (std::size_t k = 0; k < a.points.size(); ++k) {
            require(cv::norm(b.points[k] - a.points[k] - delta) < 0.002, "piece shape changed");
            const auto p = b.points[k];
            require(p.x >= workspace.x - 0.002 && p.y >= workspace.y - 0.002 &&
                    p.x <= workspace.br().x + 0.002 && p.y <= workspace.br().y + 0.002, "outside workspace");
            // 独立检查每条边的采样点到其他整个多边形的有符号距离。
            for (std::size_t j = 0; j < saved.size(); ++j) {
                if (i == j) continue;
                for (int sample = 0; sample <= 100; ++sample) {
                    const float t = sample / 100.0f;
                    const auto edge_point = p * (1 - t) + b.points[(k + 1) % b.points.size()] * t;
                    require(cv::pointPolygonTest(machine.placements[j].points, edge_point, true) <= -39.998,
                            "pieces overlap or edge gap is less than 10 mm");
                }
            }
        }
    }
    if (saved.size() > 1)
        require(machine.minimum_gap_pixels >= 40 && machine.minimum_gap_pixels < 40.1, "incorrect minimum gap");
}
}

int main() {
    try {
        const cv::Rect2f paper(0, 0, 840, 1188);
        verify(layout({{{240,240},{440,240},{440,360},{240,360}},
                       {{440,240},{640,240},{640,360},{440,360}},
                       {{240,360},{440,360},{440,480},{240,480}},
                       {{440,360},{640,360},{640,480},{440,480}}}), paper);
        // 斜接缝及三个拼块公共顶点，不能只按 x/y 各挪 10 mm。
        verify(layout({{{200,200},{600,200},{380,350}},
                       {{600,200},{600,500},{380,350}},
                       {{600,500},{200,500},{200,200},{380,350}}}), paper);
        // 凹形块及其凹槽内的矩形。
        verify(layout({{{0,0},{400,0},{400,120},{200,120},{200,240},{0,240}},
                       {{200,120},{400,120},{400,240},{200,240}}}), paper);
        verify(layout({{{0,0},{400,0},{400,240},{0,240}}}), paper);
        const auto pair = layout({{{0,0},{200,0},{200,200},{0,200}},
                                  {{200,0},{400,0},{400,200},{200,200}}});
        auto rejected = make_machine_result(pair, 40, {0,0,400,240});
        require(!rejected.ready && rejected.placements.empty(), "out-of-bounds layout accepted");
        require(!make_machine_result(PuzzleResult{}, 40, paper).ready, "unsolved puzzle accepted");
        require(!make_machine_result(pair, -1, paper).ready, "negative gap accepted");
        require(!make_machine_result(pair, std::numeric_limits<double>::quiet_NaN(), paper).ready, "NaN gap accepted");
        auto coincident = pair;
        coincident.placements[1].center_gravity = coincident.placements[0].center_gravity;
        require(!make_machine_result(coincident, 40, paper).ready, "coincident centers accepted");
        std::cout << "machine result tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
