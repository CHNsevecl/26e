#include "puzzle_solver.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using Polygon = std::vector<cv::Point2f>;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

Polygon move(Polygon points, double angle, cv::Point2f offset) {
    for (auto& p : points) {
        const auto original = p;
        p = {static_cast<float>(std::cos(angle) * original.x - std::sin(angle) * original.y) + offset.x,
             static_cast<float>(std::sin(angle) * original.x + std::cos(angle) * original.y) + offset.y};
    }
    return points;
}

PuzzleOptions exact_options() {
    PuzzleOptions options;
    options.position_tolerance = 0.02;
    options.angle_tolerance_degrees = 0.02;
    options.min_coverage = 0.999;
    return options;
}

void verify(const std::vector<Polygon>& input, const PuzzleResult& result, cv::Rect2f target, bool convex) {
    require(result.solved, "expected a complete solution");
    std::vector<bool> found(input.size(), false);
    for (const auto& placement : result.placements) {
        require(placement.source_index < input.size(), "invalid source index");
        require(!found[placement.source_index], "piece used twice");
        found[placement.source_index] = true;
        const auto& source = input[placement.source_index];
        require(placement.points.size() == source.size(), "vertex count changed");
        // 从原轮廓按鞋带公式求重心，再验证刚性变换后的重心位置。
        double twice_area = 0.0, weighted_x = 0.0, weighted_y = 0.0;
        for (std::size_t i = 0; i < source.size(); ++i) {
            const auto a = source[i], b = source[(i + 1) % source.size()];
            const double weight = static_cast<double>(a.x) * b.y - static_cast<double>(a.y) * b.x;
            twice_area += weight;
            weighted_x += (static_cast<double>(a.x) + b.x) * weight;
            weighted_y += (static_cast<double>(a.y) + b.y) * weight;
        }
        const double cx = weighted_x / (3.0 * twice_area), cy = weighted_y / (3.0 * twice_area);
        const double angle = placement.rotation_radians;
        const cv::Point2f expected_center(
            std::cos(angle) * cx - std::sin(angle) * cy + placement.translation.x,
            std::sin(angle) * cx + std::cos(angle) * cy + placement.translation.y);
        require(cv::norm(placement.center_gravity - expected_center) < 0.01, "final center of gravity is incorrect");
        for (std::size_t i = 0; i < source.size(); ++i) {
            const auto p = placement.points[i];
            require(p.x >= target.x - 0.03 && p.x <= target.x + target.width + 0.03 &&
                    p.y >= target.y - 0.03 && p.y <= target.y + target.height + 0.03, "piece outside frame");
            // 检查任意两顶点距离，而不只是求解器报告的误差值，禁止缩放、剪裁或改变形状。
            for (std::size_t j = 0; j < source.size(); ++j)
                require(std::abs(cv::norm(p - placement.points[j]) - cv::norm(source[i] - source[j])) < 0.01,
                        "rigid shape was changed");
        }
    }
    for (std::size_t i = 0; i < input.size(); ++i)
        require(found[i] == !input[i].empty(), "missing piece or empty contour treated as piece");
    // 独立采样验证外框内部被覆盖，包括凹槽；不依赖求解器自身的覆盖率统计。
    for (float y = target.y + 2.37f; y < target.y + target.height; y += 5.0f)
        for (float x = target.x + 1.73f; x < target.x + target.width; x += 5.0f) {
            int count = 0;
            for (const auto& placement : result.placements)
                if (cv::pointPolygonTest(placement.points, cv::Point2f(x, y), false) >= 0) ++count;
            require(count == 1, "exact tiling contains a gap or overlap");
        }
    if (convex) {
        for (std::size_t i = 0; i < result.placements.size(); ++i)
            for (std::size_t j = 0; j < i; ++j) {
                Polygon intersection;
                require(cv::intersectConvexConvex(result.placements[i].points, result.placements[j].points, intersection) < 0.1,
                        "exact pieces overlap");
            }
    }
}

void verify_measured(const std::vector<Polygon>& input, cv::Rect2f target) {
    const PuzzleOptions options;
    const auto result = solve_puzzle(input, target, options);
    require(result.solved && result.placements.size() == input.size(), "photographed puzzle regression failed");
    require(result.coverage >= options.min_coverage &&
            result.overlap_area <= target.area() * options.max_overlap_ratio &&
            result.outside_area <= target.area() * options.max_outside_ratio,
            "photographed solution exceeded unchanged global tolerances");
    double source_area = 0.0;
    for (const auto& polygon : input) source_area += std::abs(cv::contourArea(polygon));
    require(std::abs(source_area - (result.coverage * target.area() + result.overlap_area + result.outside_area)) < target.area() * 0.0015,
            "raster statistics inflated polygon area with boundary pixels");
    std::vector<bool> found(input.size(), false);
    for (const auto& placement : result.placements) {
        require(placement.source_index < input.size() && !found[placement.source_index], "photographed piece missing or repeated");
        found[placement.source_index] = true;
        const auto expected = move(input[placement.source_index], placement.rotation_radians, placement.translation);
        const auto moments = cv::moments(input[placement.source_index]);
        const auto center = move({cv::Point2f(moments.m10 / moments.m00, moments.m01 / moments.m00)},
                                 placement.rotation_radians, placement.translation)[0];
        require(cv::norm(center - placement.center_gravity) < 0.02, "refined center is inconsistent with pose");
        for (std::size_t j = 0; j < placement.points.size(); ++j) {
            require(cv::norm(placement.points[j] - expected[j]) < 0.02, "refined vertices are inconsistent with pose");
            const auto p = placement.points[j];
            require(p.x >= target.x - options.position_tolerance && p.x <= target.x + target.width + options.position_tolerance &&
                    p.y >= target.y - options.position_tolerance && p.y <= target.y + target.height + options.position_tolerance,
                    "refinement exceeded boundary tolerance");
        }
    }
}
} // namespace

int main() {
    try {
        const cv::Rect2f target(240, 240, 400, 240);
        const auto options = exact_options();
        std::vector<Polygon> pieces = {
            {{0, 0}, {0, 240}, {320, 240}},
            {{0, 0}, {96, 72}, {400, 120}, {400, 0}},
            {{96, 72}, {400, 120}, {400, 152}, {256, 192}},
            {{256, 192}, {320, 240}, {400, 240}, {400, 152}},
            {} // 模拟识别时被过滤的小轮廓。
        };
        for (std::size_t i = 0; i < pieces.size(); ++i)
            pieces[i] = move(pieces[i], 0.37 * (i + 1), {static_cast<float>(i * 73), static_cast<float>(i * 91)});
        const auto original = pieces;
        const auto four = solve_puzzle(pieces, target, options);
        verify(pieces, four, target, true);
        require(pieces == original, "search mutated input");

        // 实拍三块回归：长接缝约 3 像素的偏差，曾被“较小拼块面积的 2%”误拒绝。
        const std::vector<Polygon> photographed_three = {
            {{691.7702f, 851.2958f}, {420.2774f, 1018.0f}, {660.7446f, 1056.5729f}},
            {{64.4033f, 917.6425f}, {339.3846f, 1143.6920f}, {333.6706f, 828.0518f}},
            {{311.6704f, 648.5903f}, {305.2354f, 694.9279f}, {533.0499f, 872.0728f}, {552.7147f, 677.6553f}}
        };
        // 同一套三块换方向后，必须允许兼顾外框和接缝的折中角度。
        const std::vector<Polygon> photographed_three_rotated = {
            {{456.5792f, 1053.6160f}, {742.7378f, 914.2788f}, {507.7045f, 852.9472f}},
            {{92.4827f, 750.8419f}, {78.7723f, 1066.1715f}, {355.9521f, 990.1800f}},
            {{476.7284f, 842.1863f}, {432.7982f, 651.2646f}, {196.5785f, 701.7726f}, {205.4528f, 747.3037f}}
        };
        const std::vector<Polygon> photographed_four = {
            {{358.8131f, 899.9789f}, {280.4814f, 922.8569f}, {236.1366f, 987.2281f}, {301.5154f, 1035.6433f}},
            {{370.1721f, 677.6198f}, {351.7906f, 796.3520f}, {448.4930f, 1088.0764f}, {556.1565f, 1032.0745f}},
            {{294.8371f, 596.2952f}, {58.8731f, 636.8846f}, {119.6016f, 954.0363f}},
            {{442.1013f, 584.3672f}, {424.2782f, 609.3625f}, {646.3202f, 824.4530f}, {581.0823f, 639.6201f}}
        };
        // 第三次实拍：面积偏大且姿态不同，旧深搜和包含边界像素的面积统计曾导致超时。
        const std::vector<Polygon> photographed_three_third = {
            {{409.5856f, 806.4825f}, {379.3143f, 842.8383f}, {483.9674f, 1109.2220f}, {601.0884f, 955.1001f}},
            {{114.6690f, 863.7070f}, {433.7686f, 713.3890f}, {145.5042f, 579.2198f}},
            {{444.2116f, 526.3265f}, {495.9233f, 767.6967f}, {699.5896f, 722.7185f}}
        };
        for (const auto& photo : {photographed_three, photographed_three_rotated, photographed_four, photographed_three_third}) {
            verify_measured(photo, target);
            // 旋转、平移各个源拼块后，求解与输出姿态仍须一致。
            for (int variation = 1; variation <= 3; ++variation) {
                auto moved = photo;
                for (std::size_t i = 0; i < moved.size(); ++i)
                    moved[i] = move(moved[i], variation * 0.73 * (i + 1), {37.0f * variation, -53.0f * variation});
                verify_measured(moved, target);
            }
        }
        // 模拟亚像素识别随拍摄变化的轻微扰动，不能只对固定顶点集合有效。
        for (int variation = 1; variation <= 4; ++variation) {
            auto noisy = photographed_three_third;
            for (std::size_t i = 0; i < noisy.size(); ++i)
                for (std::size_t j = 0; j < noisy[i].size(); ++j) {
                    const double phase = variation * 13 + i * 11 + j * 3;
                    noisy[i][j].x += 0.35 * std::sin(phase);
                    noisy[i][j].y += 0.35 * std::cos(phase + 0.7);
                }
            verify_measured(noisy, target);
        }

        // 一个恰好铺满外框的矩形，统计中不能凭空增加框外边界面积。
        const auto rectangle = solve_puzzle({{{0, 0}, {400, 0}, {400, 240}, {0, 240}}}, target);
        require(rectangle.solved && rectangle.coverage == 1.0 && rectangle.overlap_area == 0.0 && rectangle.outside_area == 0.0,
                "exact frame acquired spurious boundary area");

        // 第二题尺寸范围：包括上下限、不同长宽比，以及与 400×240 等面积的 480×200。
        const cv::Size2f min_size(360, 200), max_size(480, 360);
        for (const auto size : {cv::Size2f(360, 200), cv::Size2f(480, 360), cv::Size2f(440, 280),
                                cv::Size2f(480, 200), cv::Size2f(454, 286)}) {
            const float w = size.width, h = size.height;
            std::vector<Polygon> varied = {
                {{0, 0}, {0.25f * w, 0}, {0.65f * w, h}, {0, h}},
                {{0.25f * w, 0}, {w, 0}, {0.65f * w, h}},
                {{0.65f * w, h}, {w, 0}, {w, h}}
            };
            for (std::size_t i = 0; i < varied.size(); ++i)
                varied[i] = move(varied[i], 0.53 * (i + 1), {73.0f * i, 91.0f * i});
            const auto unchanged = varied;
            const auto automatic = solve_puzzle_in_range(varied, target.tl(), min_size, max_size, options);
            require(automatic.solved, "automatic rectangle size search failed");
            require(std::abs(automatic.target.width - w) < 1.0 && std::abs(automatic.target.height - h) < 1.0,
                    "automatic size inferred wrong aspect ratio");
            require(automatic.target.width >= min_size.width && automatic.target.width <= max_size.width &&
                    automatic.target.height >= min_size.height && automatic.target.height <= max_size.height,
                    "automatic size escaped allowed range");
            verify(varied, automatic, automatic.target, true);
            require(varied == unchanged, "automatic size search mutated input");
        }
        for (const auto& photo : {photographed_three_third, photographed_four}) {
            const auto automatic = solve_puzzle_in_range(photo, target.tl(), min_size, max_size);
            require(automatic.solved && automatic.placements.size() == photo.size(), "automatic photographed puzzle failed");
            require(automatic.target.width >= min_size.width && automatic.target.width <= max_size.width &&
                    automatic.target.height >= min_size.height && automatic.target.height <= max_size.height,
                    "photographed target escaped size range");
        }
        require(!solve_puzzle_in_range(pieces, target.tl(), max_size, min_size).solved, "reversed size range accepted");
        const auto oversized = solve_puzzle_in_range({{{0, 0}, {5000, 0}, {5000, 5000}, {0, 5000}}}, target.tl(), min_size, max_size);
        require(!oversized.solved && oversized.target.empty(), "out-of-range area reported as solved");
        // 测试范围允许宽 70–120 mm：这组三块约为 80×70 mm，原 90 mm 下限会排除该摆法。
        const cv::Size2f test_min_size(280, 200);
        const std::vector<Polygon> smaller_photographed = {
            {{707.097656f, 882.831055f}, {356.234283f, 996.587769f}, {669.482056f, 1055.55554f}},
            {{343.684235f, 860.963013f}, {279.335846f, 726.985657f}, {25.1342506f, 843.123779f}, {170.085495f, 945.220642f}},
            {{524.183655f, 644.138733f}, {365.614136f, 708.384888f}, {446.3862f, 886.19458f}, {565.477905f, 738.042908f}}
        };
        const auto smaller_original = smaller_photographed;
        const auto smaller = solve_puzzle_in_range(smaller_photographed, target.tl(), test_min_size, max_size);
        require(smaller.solved && smaller.placements.size() == 3, "smaller photographed puzzle failed in test range");
        require(std::abs(smaller.target.width - 320) < 8 && std::abs(smaller.target.height - 280) < 8,
                "smaller photographed puzzle has unexpected dimensions");
        require(smaller_photographed == smaller_original, "test-range solver changed source polygons");
        for (const auto& placement : smaller.placements) {
            require(placement.source_index < smaller_photographed.size(), "test-range source index invalid");
            const auto transformed = move(smaller_photographed[placement.source_index], placement.rotation_radians, placement.translation);
            for (std::size_t i = 0; i < transformed.size(); ++i)
                require(cv::norm(transformed[i] - placement.points[i]) < 0.02, "test-range pose changed shape");
        }
        for (const auto size : {cv::Size2f(280, 200), cv::Size2f(320, 280)}) {
            std::vector<Polygon> split = {
                {{0, 0}, {size.width, 0}, {0, size.height}},
                {{size.width, 0}, {size.width, size.height}, {0, size.height}}
            };
            const auto automatic = solve_puzzle_in_range(split, target.tl(), test_min_size, max_size, options);
            require(automatic.solved && std::abs(automatic.target.width - size.width) < 1 &&
                    std::abs(automatic.target.height - size.height) < 1, "expanded-range size regression failed");
            verify(split, automatic, automatic.target, true);
        }
        for (const auto& photo : {photographed_three, photographed_three_rotated, photographed_three_third, photographed_four}) {
            const auto automatic = solve_puzzle_in_range(photo, target.tl(), test_min_size, max_size);
            require(automatic.solved && automatic.placements.size() == photo.size(), "expanded range regressed previous photograph");
        }
        auto auto_budget = options;
        auto_budget.max_nodes = 1;
        const auto auto_limited = solve_puzzle_in_range(pieces, target.tl(), min_size, max_size, auto_budget);
        require(!auto_limited.solved && auto_limited.search_limit_reached && auto_limited.explored_nodes == 1,
                "size candidates did not share search budget");

        // 大矩形先以错误朝向放入时，400 × 60 的长条无法继续放置，必须回退。
        const std::vector<Polygon> needs_backtracking = {
            {{0, 0}, {0, 180}, {200, 180}, {200, 0}},
            {{200, 0}, {200, 180}, {400, 180}, {400, 0}},
            {{0, 180}, {400, 180}, {400, 240}, {0, 240}}
        };
        const auto retried = solve_puzzle(needs_backtracking, target, options);
        verify(needs_backtracking, retried, target, true);
        require(retried.explored_nodes > needs_backtracking.size() + 1, "backtracking case did not exercise a failed branch");

        // 内部块不接触外框，需要利用已放拼块的边生成候选。
        std::vector<Polygon> interior = {
            {{0, 0}, {400, 0}, {400, 60}, {0, 60}},
            {{0, 180}, {400, 180}, {400, 240}, {0, 240}},
            {{0, 60}, {100, 60}, {100, 180}, {0, 180}},
            {{300, 60}, {400, 60}, {400, 180}, {300, 180}},
            {{100, 60}, {300, 60}, {300, 180}, {100, 180}}
        };
        verify(interior, solve_puzzle(interior, target, options), target, true);

        // 凹多边形不能用凸包碰撞检测，否则会把凹槽误判成已占用。
        std::vector<Polygon> concave = {
            {{0, 0}, {400, 0}, {400, 120}, {200, 120}, {200, 240}, {0, 240}},
            {{200, 120}, {400, 120}, {400, 240}, {200, 240}}
        };
        verify(concave, solve_puzzle(concave, target, options), target, false);

        // 面积相同也不一定能放进外框；失败不返回未确认的候选。
        const auto impossible = solve_puzzle({{{0, 0}, {800, 0}, {800, 120}, {0, 120}}}, target, options);
        require(!impossible.solved && impossible.placements.empty(), "impossible puzzle reported as solved");
        require(!solve_puzzle({}, target).solved, "empty input reported as solved");
        require(!solve_puzzle({{{0, 0}, {10, 0}, {0, 10}}}, target).solved, "insufficient area accepted");
        require(!solve_puzzle({{{0, 0}, {std::numeric_limits<float>::quiet_NaN(), 0}, {0, 10}}}, target).solved,
                "non-finite input accepted");
        auto limited_options = options;
        limited_options.max_nodes = 1;
        const auto limited = solve_puzzle(pieces, target, limited_options);
        require(!limited.solved && limited.search_limit_reached && limited.placements.empty(), "search budget ignored");
        std::cout << "PASS: exact dissection, backtracking, interior piece, concave piece, invalid/unsolvable input, budget, immutable input\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
