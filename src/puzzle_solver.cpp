#include "puzzle_solver.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <map>
#include <limits>
#include <set>
#include <tuple>

namespace {
using Polygon = std::vector<cv::Point2f>;
using Triangle = std::array<cv::Point2f, 3>;

bool valid_options(const PuzzleOptions& options) {
    return std::isfinite(options.position_tolerance) && options.position_tolerance >= 0 &&
        std::isfinite(options.angle_tolerance_degrees) && options.angle_tolerance_degrees >= 0 &&
        std::isfinite(options.min_coverage) && options.min_coverage > 0 && options.min_coverage <= 1 &&
        std::isfinite(options.max_overlap_ratio) && options.max_overlap_ratio >= 0 &&
        std::isfinite(options.max_outside_ratio) && options.max_outside_ratio >= 0 &&
        std::isfinite(options.time_limit_seconds) && options.time_limit_seconds > 0;
}

double cross(cv::Point2f a, cv::Point2f b) {
    return static_cast<double>(a.x) * b.y - static_cast<double>(a.y) * b.x;
}

cv::Point2f rotate(cv::Point2f p, double angle) {
    const double c = std::cos(angle), s = std::sin(angle);
    return {static_cast<float>(c * p.x - s * p.y),
            static_cast<float>(s * p.x + c * p.y)};
}

// 耳切法：凹多边形也按真实面积检测，不能用凸包代替。
std::vector<Triangle> triangulate(Polygon polygon) {
    if (cv::contourArea(polygon, true) < 0) std::reverse(polygon.begin(), polygon.end());
    std::vector<Triangle> triangles;
    while (polygon.size() > 3) {
        bool clipped = false;
        for (std::size_t i = 0; i < polygon.size(); ++i) {
            const auto a = polygon[(i + polygon.size() - 1) % polygon.size()];
            const auto b = polygon[i], c = polygon[(i + 1) % polygon.size()];
            if (cross(b - a, c - b) <= 1e-6) continue;
            bool contains = false;
            for (std::size_t j = 0; j < polygon.size(); ++j) {
                if (j == i || j == (i + 1) % polygon.size() ||
                    j == (i + polygon.size() - 1) % polygon.size()) continue;
                const auto p = polygon[j];
                if (cross(b - a, p - a) >= -1e-6 &&
                    cross(c - b, p - b) >= -1e-6 &&
                    cross(a - c, p - c) >= -1e-6) {
                    contains = true;
                    break;
                }
            }
            if (contains) continue;
            triangles.push_back({a, b, c});
            polygon.erase(polygon.begin() + i);
            clipped = true;
            break;
        }
        if (!clipped) return {};
    }
    if (polygon.size() == 3 && std::abs(cross(polygon[1] - polygon[0], polygon[2] - polygon[0])) > 1e-6)
        triangles.push_back({polygon[0], polygon[1], polygon[2]});
    return triangles;
}

struct Shape {
    std::size_t source_index;
    Polygon points;
    std::vector<Triangle> triangles;
    double area;
};

struct Candidate {
    PuzzlePlacement placement;
    std::vector<Triangle> triangles;
    double score = 0.0;
    double overlap = 0.0;
    double outside = 0.0;
};

// 按像素中心采样填充，不使用 fillPoly 的“包含轮廓像素”语义。
// 否则每条接缝会额外算半个像素宽，拼块数量/方向变化也会改变虚增的面积。
cv::Mat sample_polygon(const Polygon& polygon, cv::Size size, cv::Point2f origin, int sampling_scale) {
    cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
    std::vector<double> crossings;
    for (int row = 0; row < size.height; ++row) {
        const double y = origin.y + (row + 0.5) / sampling_scale;
        crossings.clear();
        for (std::size_t i = 0; i < polygon.size(); ++i) {
            const auto a = polygon[i], b = polygon[(i + 1) % polygon.size()];
            if ((a.y <= y && y < b.y) || (b.y <= y && y < a.y))
                crossings.push_back(a.x + (y - a.y) * (b.x - a.x) / (b.y - a.y));
        }
        std::sort(crossings.begin(), crossings.end());
        for (std::size_t i = 1; i < crossings.size(); i += 2) {
            const int left = std::clamp(static_cast<int>(std::ceil((crossings[i - 1] - origin.x) * sampling_scale - 0.5)), 0, size.width);
            const int right = std::clamp(static_cast<int>(std::ceil((crossings[i] - origin.x) * sampling_scale - 0.5)), 0, size.width);
            if (left < right) mask.row(row).colRange(left, right).setTo(255);
        }
    }
    return mask;
}

double intersection_area(const std::vector<Triangle>& a, const std::vector<Triangle>& b) {
    double area = 0.0;
    Polygon intersection;
    for (const auto& ta : a) {
        const Polygon pa(ta.begin(), ta.end());
        for (const auto& tb : b) {
            const Polygon pb(tb.begin(), tb.end());
            area += std::max(0.0f, cv::intersectConvexConvex(pa, pb, intersection));
        }
    }
    return area;
}

// 近似共线且沿边有重合长度，才算接缝；仅端点靠近不算。
double contact_length(const Polygon& a, const Polygon& b, const PuzzleOptions& options) {
    double length = 0.0;
    const double sine_limit = std::sin(options.angle_tolerance_degrees * CV_PI / 180.0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto p = a[i], v = a[(i + 1) % a.size()] - p;
        const double edge_length = cv::norm(v);
        if (edge_length < 1e-6) continue;
        const auto direction = v * (1.0 / edge_length);
        for (std::size_t j = 0; j < b.size(); ++j) {
            const auto q = b[j], r = b[(j + 1) % b.size()];
            const double other_length = cv::norm(r - q);
            if (other_length < 1e-6 ||
                std::abs(cross(v, r - q)) > sine_limit * edge_length * other_length) continue;
            if (std::max(std::abs(cross(direction, q - p)),
                         std::abs(cross(direction, r - p))) > options.position_tolerance) continue;
            const double x = direction.dot(q - p), y = direction.dot(r - p);
            length += std::max(0.0, std::min(edge_length, std::max(x, y)) -
                                    std::max(0.0, std::min(x, y)));
        }
    }
    return length;
}

class Search {
public:
    Search(const std::vector<Polygon>& polygons, cv::Rect2f target, const PuzzleOptions& options,
           std::vector<cv::Rect2f> targets = {}, cv::Size2f min_size = {}, cv::Size2f max_size = {})
        : target_(target), options_(options), start_(std::chrono::steady_clock::now()),
          targets_(std::move(targets)), min_size_(min_size), max_size_(max_size) {
        set_target(target);
        if (targets_.empty()) result_.target = target;
        for (std::size_t i = 0; i < polygons.size(); ++i) {
            if (polygons[i].empty()) continue; // line_process 会为被过滤的轮廓保留空槽。
            for (const auto p : polygons[i])
                if (!std::isfinite(p.x) || !std::isfinite(p.y)) { valid_ = false; return; }
            auto triangles = triangulate(polygons[i]);
            const double area = std::abs(cv::contourArea(polygons[i]));
            if (triangles.empty() || area < 1e-3) { valid_ = false; return; }
            shapes_.push_back({i, polygons[i], std::move(triangles), area});
        }
        std::stable_sort(shapes_.begin(), shapes_.end(), [](const Shape& a, const Shape& b) {
            return a.area > b.area;
        });
        used_.resize(shapes_.size(), false);
    }

    PuzzleResult run() {
        if (!valid_ || shapes_.empty()) return result_;
        double total_area = 0.0;
        for (const auto& shape : shapes_) total_area += shape.area;
        const double target_area = target_.area();
        // 面积已经不可能满足最终验收时，不开始穷举。
        if (total_area < target_area * options_.min_coverage ||
            total_area > target_area * (1.0 + options_.max_overlap_ratio + options_.max_outside_ratio))
            return result_;
        // 先并行保留多套局部布局，再联合拟合；不能在一个近似朝向上深搜到超时。
        search_beam();
        if (!result_.solved && !result_.search_limit_reached) {
            placed_.clear();
            for (const auto target : targets_.empty() ? std::vector<cv::Rect2f>{target_} : targets_) {
                set_target(target);
                if (visit() || result_.search_limit_reached) break;
            }
        }
        if (result_.solved && uses_appearance()) {
            improve_exchanges();
            improve_appearance();
        }
        return result_;
    }

private:
    double appearance_score(const std::vector<Candidate>& layout) const {
        double score = 0;
        if (options_.seam_score)
            for (std::size_t i = 0; i < layout.size(); ++i)
                for (std::size_t j = 0; j < i; ++j) {
                    const double value = options_.seam_score(layout[i].placement, layout[j].placement);
                    if (std::isfinite(value)) score += value;
                }
        return score;
    }

    double boundary_score(const std::vector<Candidate>& layout) const {
        double score = 0;
        if (options_.boundary_score)
            for (const auto& candidate : layout) {
                const double value = options_.boundary_score(candidate.placement, target_);
                if (std::isfinite(value)) score += value;
            }
        return score;
    }

    PuzzleLayoutScore layout_score(const std::vector<Candidate>& layout) const {
        if (!options_.layout_score) return {};
        std::vector<PuzzlePlacement> placements;
        for (const auto& candidate : layout) placements.push_back(candidate.placement);
        auto score=options_.layout_score(placements,target_);
        if (!std::isfinite(score.score)) score.score=0;
        return score;
    }

    bool uses_appearance() const { return options_.seam_score || options_.boundary_score || options_.layout_score; }

    void set_target(cv::Rect2f target) {
        target_ = target;
        border_ = {{target.x, target.y}, {target.x + target.width, target.y},
                   {target.x + target.width, target.y + target.height}, {target.x, target.y + target.height}};
        border_triangles_ = triangulate(border_);
    }

    bool limit_reached() {
        if (result_.explored_nodes >= options_.max_nodes ||
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count() >= options_.time_limit_seconds) {
            result_.search_limit_reached = true;
            return true;
        }
        return false;
    }

    std::vector<Candidate> candidates(const Shape& shape, bool seed_layout = false) {
        std::vector<Candidate> candidates;
        std::set<std::tuple<long long, long long, long long>> seen;
        std::vector<Polygon> references{border_};
        for (const auto& placed : placed_) references.push_back(placed.placement.points);
        auto construction_options = options_;
        if (seed_layout) construction_options.position_tolerance *= 2.0;
        const double tolerance = construction_options.position_tolerance;
        const double construction_factor = seed_layout ? 4.0 : 1.0;
        // 收集单边对齐角度。相邻约束因识别误差略有冲突时，还需尝试两者的折中角度。
        std::set<long long> alignment_keys;
        std::vector<double> alignments;
        for (std::size_t edge = 0; edge < shape.points.size(); ++edge) {
            const auto v = shape.points[(edge + 1) % shape.points.size()] - shape.points[edge];
            if (cv::norm(v) < 1e-6) continue;
            for (const auto& reference : references) {
                for (std::size_t j = 0; j < reference.size(); ++j) {
                    const auto w = reference[(j + 1) % reference.size()] - reference[j];
                    if (cv::norm(w) < 1e-6) continue;
                    for (int reverse = 0; reverse < 2; ++reverse) {
                        const double angle = std::remainder(std::atan2(w.y, w.x) - std::atan2(v.y, v.x) + reverse * CV_PI, 2 * CV_PI);
                        if (alignment_keys.insert(std::llround(angle * 10000)).second) alignments.push_back(angle);
                    }
                }
            }
        }
        for (std::size_t edge = 0; edge < shape.points.size(); ++edge) {
            const auto a = shape.points[edge], b = shape.points[(edge + 1) % shape.points.size()];
            if (cv::norm(b - a) < 1e-6) continue;
            const double source_angle = std::atan2(b.y - a.y, b.x - a.x);
            for (const auto& reference : references) {
                for (std::size_t j = 0; j < reference.size(); ++j) {
                    const auto p = reference[j], q = reference[(j + 1) % reference.size()];
                    const double reference_angle = std::atan2(q.y - p.y, q.x - p.x);
                    for (int reverse = 0; reverse < 2; ++reverse) {
                        const double base_angle = std::remainder(reference_angle - source_angle + reverse * CV_PI, 2 * CV_PI);
                        std::vector<double> angles{base_angle};
                        for (const double other : alignments) {
                            const double difference = std::remainder(other - base_angle, 2 * CV_PI);
                            if (std::abs(difference) > 1e-4 &&
                                std::abs(difference) <= options_.angle_tolerance_degrees * CV_PI / 180.0)
                                angles.push_back(std::remainder(base_angle + difference * 0.5, 2 * CV_PI));
                        }
                        for (const double angle : angles) {
                            // 两端对齐覆盖长边由多块拼成的情况；中心对齐吸收测量误差。
                            const std::array<cv::Point2f, 3> sources{a, b, (a + b) * 0.5f};
                            const std::array<cv::Point2f, 3> targets{p, q, (p + q) * 0.5f};
                            for (std::size_t s = 0; s < sources.size(); ++s) {
                                for (std::size_t t = 0; t < targets.size(); ++t) {
                                    if ((s == 2) != (t == 2)) continue;
                                    if (limit_reached()) return candidates;
                                    const auto translation = targets[t] - rotate(sources[s], angle);
                                    const auto key = std::make_tuple(std::llround(angle * 10000),
                                        std::llround(translation.x * 4), std::llround(translation.y * 4));
                                    if (!seen.insert(key).second) continue;
                                    Candidate candidate;
                                    candidate.placement = {shape.source_index, angle, translation, {}};
                                    bool outside = false;
                                    for (const auto point : shape.points) {
                                        const auto transformed = rotate(point, angle) + translation;
                                        candidate.placement.points.push_back(transformed);
                                        if (transformed.x < target_.x - tolerance ||
                                            transformed.x > target_.x + target_.width + tolerance ||
                                            transformed.y < target_.y - tolerance ||
                                            transformed.y > target_.y + target_.height + tolerance) outside = true;
                                    }
                                    if (outside) continue;
                                    for (const auto& triangle : shape.triangles) {
                                        Triangle transformed;
                                        for (std::size_t k = 0; k < 3; ++k) transformed[k] = rotate(triangle[k], angle) + translation;
                                        candidate.triangles.push_back(transformed);
                                    }
                                    candidate.outside = std::max(0.0, shape.area - intersection_area(candidate.triangles, border_triangles_));
                                    if (candidate.outside > construction_factor * target_.area() * options_.max_outside_ratio) continue;
                                    double contact = contact_length(candidate.placement.points, border_, construction_options);
                                    bool collision = false;
                                    for (const auto& placed : placed_) {
                                        const double overlap = intersection_area(candidate.triangles, placed.triangles);
                                        const double shared_contact = contact_length(candidate.placement.points, placed.placement.points, construction_options);
                                        // 接缝误差随接缝长度增长，不能按较小拼块面积的固定比例提前淘汰。
                                        // 允许一条 tolerance 宽的接缝带，以及角点处 tolerance² 的误差。
                                        // 全部拼块的总重叠仍由 accept_complete 的全局上限验收。
                                        const double seam_budget = tolerance * shared_contact + tolerance * tolerance;
                                        if (overlap > std::min(seam_budget, construction_factor * target_.area() * options_.max_overlap_ratio)) {
                                            collision = true;
                                            break;
                                        }
                                        candidate.overlap += overlap;
                                        contact += shared_contact;
                                    }
                                    if (collision || contact < 1.0) continue;
                                    candidate.score = contact - 0.5 * (candidate.overlap + candidate.outside);
                                    if (options_.seam_score)
                                        for (const auto& placed : placed_) {
                                            const double value = options_.seam_score(candidate.placement, placed.placement);
                                            if (std::isfinite(value)) candidate.score += value;
                                        }
                                    if (options_.boundary_score) {
                                        const double value = options_.boundary_score(candidate.placement, target_);
                                        if (std::isfinite(value)) candidate.score += value;
                                    }
                                    candidates.push_back(std::move(candidate));
                                }
                            }
                        }
                    }
                }
            }
        }
        std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.score > b.score;
        });
        return candidates;
    }

    void search_beam() {
        struct Layout { std::vector<Candidate> pieces; double score = 0.0; cv::Rect2f target; };
        const std::size_t beam_width = std::max<std::size_t>(32, targets_.size() * 4);
        constexpr std::size_t choices_per_piece = 16;
        std::vector<Layout> beam;
        for (const auto target : targets_.empty() ? std::vector<cv::Rect2f>{target_} : targets_)
            beam.push_back({{}, 0.0, target});
        for (std::size_t depth = 0; depth < shapes_.size(); ++depth) {
            std::vector<Layout> next;
            for (const auto& layout : beam) {
                if (limit_reached()) return;
                ++result_.explored_nodes;
                set_target(layout.target);
                placed_ = layout.pieces;
                for (const auto& shape : shapes_) {
                    if (std::any_of(placed_.begin(), placed_.end(), [&](const Candidate& p) {
                        return p.placement.source_index == shape.source_index;
                    })) continue;
                    auto choices = candidates(shape, true);
                    if (result_.search_limit_reached) return;
                    std::vector<std::size_t> selected;
                    for (std::size_t i = 0; i < choices.size(); ++i) {
                        // 相近姿态交给连续拟合，避免整个束都被同一角落的微小角度变体占满。
                        bool duplicate = false;
                        for (const auto old : selected) {
                            double distance = 0.0;
                            for (std::size_t j = 0; j < shape.points.size(); ++j)
                                distance = std::max(distance, cv::norm(choices[i].placement.points[j] - choices[old].placement.points[j]));
                            if (distance < std::max(0.02, options_.position_tolerance * 0.5)) { duplicate = true; break; }
                        }
                        if (duplicate) continue;
                        selected.push_back(i);
                        Layout expanded = layout;
                        expanded.score += shape.area + 2.0 * choices[i].score;
                        expanded.pieces.push_back(choices[i]);
                        next.push_back(std::move(expanded));
                        if (selected.size() == choices_per_piece) break;
                    }
                }
            }
            if (next.empty()) return;
            if (depth + 1 == shapes_.size()) {
                for (auto& layout : next) {
                    set_target(layout.target);
                    placed_ = layout.pieces;
                    layout.score = -layout_error();
                    if (uses_appearance())
                        layout.score += (appearance_score(placed_) + boundary_score(placed_)) /
                            std::max(1.0f, target_.width + target_.height);
                }
            }
            std::stable_sort(next.begin(), next.end(), [](const Layout& a, const Layout& b) { return a.score > b.score; });
            beam.clear();
            std::set<std::vector<long long>> seen;
            std::map<std::pair<int, int>, std::size_t> size_counts;
            for (auto& layout : next) {
                std::vector<const PuzzlePlacement*> ordered;
                for (const auto& p : layout.pieces) ordered.push_back(&p.placement);
                std::sort(ordered.begin(), ordered.end(), [](const auto* a, const auto* b) { return a->source_index < b->source_index; });
                const auto size_key = std::make_pair(cvRound(layout.target.width * 2), cvRound(layout.target.height * 2));
                if (!targets_.empty() && depth + 1 < shapes_.size() && size_counts[size_key] >= 4) continue;
                std::vector<long long> key{size_key.first, size_key.second};
                for (const auto* p : ordered) {
                    key.insert(key.end(), {static_cast<long long>(p->source_index), std::llround(p->rotation_radians * 1000),
                        std::llround(p->translation.x * 2), std::llround(p->translation.y * 2)});
                }
                if (!seen.insert(key).second) continue;
                ++size_counts[size_key];
                beam.push_back(std::move(layout));
                if (beam.size() == beam_width) break;
            }
            if (depth + 1 == shapes_.size()) {
                for (const auto& layout : beam) {
                    // Once a valid geometry exists, reserve budget for appearance
                    // exchanges instead of spending it all on near-duplicate fits.
                    if (uses_appearance() && shapes_.size()<=8 && result_.solved &&
                        std::chrono::duration<double>(std::chrono::steady_clock::now()-start_).count() >=
                            0.65*options_.time_limit_seconds) break;
                    if (limit_reached()) return;
                    ++result_.explored_nodes;
                    set_target(layout.target);
                    placed_ = layout.pieces;
                    // 即使初始拼法尚未通过验收，也先优化所有块的姿态，再执行原阈值验收。
                    improve_solution();
                    if (result_.solved && !uses_appearance()) return;
                }
            }
        }
    }

    bool accept_complete() {
        // 最终几何重新验证，也用于验证联合微调后的布局。
        const double tolerance = options_.position_tolerance;
        for (std::size_t i = 0; i < placed_.size(); ++i) {
            for (const auto p : placed_[i].placement.points)
                if (p.x < target_.x - tolerance || p.x > target_.x + target_.width + tolerance ||
                    p.y < target_.y - tolerance || p.y > target_.y + target_.height + tolerance) return false;
            for (std::size_t j = 0; j < i; ++j) {
                const double contact = contact_length(placed_[i].placement.points, placed_[j].placement.points, options_);
                const double overlap = intersection_area(placed_[i].triangles, placed_[j].triangles);
                if (overlap > std::min(tolerance * contact + tolerance * tolerance,
                                      target_.area() * options_.max_overlap_ratio)) return false;
            }
        }
        // 两倍分辨率计算覆盖、重复占用和框外面积；不能只检查“每块都放下了”。
        constexpr int raster_scale = 2;
        const int margin = static_cast<int>(std::ceil(options_.position_tolerance)) + 2;
        const cv::Size size(static_cast<int>(std::ceil(target_.width * raster_scale)) + 2 * margin * raster_scale,
                            static_cast<int>(std::ceil(target_.height * raster_scale)) + 2 * margin * raster_scale);
        cv::Mat occupied = cv::Mat::zeros(size, CV_8UC1);
        cv::Mat target_mask = cv::Mat::zeros(size, CV_8UC1);
        cv::rectangle(target_mask, cv::Rect(margin * raster_scale, margin * raster_scale,
            static_cast<int>(std::lround(target_.width * raster_scale)),
            static_cast<int>(std::lround(target_.height * raster_scale))), cv::Scalar(255), cv::FILLED);
        double sum_pixels = 0.0;
        for (const auto& placed : placed_) {
            const cv::Mat mask = sample_polygon(placed.placement.points, size,
                cv::Point2f(target_.x - margin, target_.y - margin), raster_scale);
            sum_pixels += cv::countNonZero(mask);
            cv::bitwise_or(occupied, mask, occupied);
        }
        cv::Mat covered, outside;
        cv::bitwise_and(occupied, target_mask, covered);
        cv::bitwise_not(target_mask, outside);
        cv::bitwise_and(occupied, outside, outside);
        const double coverage = static_cast<double>(cv::countNonZero(covered)) / cv::countNonZero(target_mask);
        const double overlap_area = (sum_pixels - cv::countNonZero(occupied)) / (raster_scale * raster_scale);
        const double outside_area = static_cast<double>(cv::countNonZero(outside)) / (raster_scale * raster_scale);
        if (coverage < options_.min_coverage || overlap_area > target_.area() * options_.max_overlap_ratio ||
            outside_area > target_.area() * options_.max_outside_ratio) return false;
        ++result_.evaluated_solutions;
        const double appearance = appearance_score(placed_);
        const double boundary = boundary_score(placed_);
        // 不用极小的颜色采样收益换取更多缺口、重叠或框外面积。
        const auto whole_layout=layout_score(placed_);
        const double quality = appearance + boundary + whole_layout.score - 0.02 * (overlap_area + outside_area +
            (1.0 - coverage) * target_.area());
        if (uses_appearance() && result_.solved &&
            ((result_.layout_accepted && !whole_layout.accepted) ||
             (result_.layout_accepted==whole_layout.accepted && quality<=best_solution_quality_))) return true;
        best_solution_quality_ = quality;
        result_.seam_score = appearance;
        result_.boundary_score = boundary;
        result_.layout_score = whole_layout.score;
        result_.layout_accepted = whole_layout.accepted;
        ++solution_revision_;
        result_.solved = true;
        result_.target = target_;
        result_.coverage = coverage;
        result_.overlap_area = overlap_area;
        result_.outside_area = outside_area;
        result_.placements.clear();
        for (const auto& placed : placed_) {
            auto placement = placed.placement;
            // 使用最终轮廓的面积重心，不能用顶点坐标的算术平均代替。
            const auto moments = cv::moments(placement.points);
            placement.center_gravity = cv::Point2f(moments.m10 / moments.m00, moments.m01 / moments.m00);
            result_.placements.push_back(std::move(placement));
        }
        return true;
    }

    double layout_error() const {
        double outside = 0.0, overlap = 0.0, boundary_error = 0.0, total_area = 0.0;
        for (std::size_t i = 0; i < placed_.size(); ++i) {
            const auto& candidate = placed_[i];
            const double area = std::abs(cv::contourArea(candidate.placement.points));
            total_area += area;
            outside += std::max(0.0, area - intersection_area(candidate.triangles, border_triangles_));
            for (std::size_t j = 0; j < i; ++j)
                overlap += intersection_area(candidate.triangles, placed_[j].triangles);
            for (const auto p : candidate.placement.points) {
                const double distance = std::max({0.0, static_cast<double>(target_.x - p.x),
                    static_cast<double>(p.x - target_.x - target_.width), static_cast<double>(target_.y - p.y),
                    static_cast<double>(p.y - target_.y - target_.height)}) - options_.position_tolerance;
                if (distance > 0) boundary_error += distance * distance;
            }
        }
        // 同时减少缺口并均衡重叠/框外误差，避免贴着某一项验收上限取解。
        const double missing = std::max(0.0, target_.area() - total_area + overlap + outside);
        const double missing_error = targets_.empty() ? 0.0 :
            missing / std::max(1e-6, target_.area() * (1.0 - options_.min_coverage));
        return (overlap + outside) / target_.area() + boundary_error + missing_error +
               std::max(overlap / std::max(1e-6, target_.area() * options_.max_overlap_ratio),
                        outside / std::max(1e-6, target_.area() * options_.max_outside_ratio));
    }

    void improve_solution() {
        const auto original = placed_;
        const auto original_target = target_;
        // 微调是有界的附加步骤；失败时保留已经确认的解，不能破坏回退状态。
        const auto deadline = std::min(start_ + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(options_.time_limit_seconds)),
            std::chrono::steady_clock::now() + std::chrono::milliseconds(250));
        const auto within_budget = [&] { return std::chrono::steady_clock::now() < deadline; };
        double best_error = layout_error();
        for (const double step : {1.0, 0.5, 0.25}) {
            if (!within_budget()) break;
            for (int pass = 0; pass < 8 && within_budget(); ++pass) {
                bool improved = false;
                for (std::size_t i = 0; i < placed_.size() && within_budget(); ++i) {
                    for (int axis = 0; axis < 3 && within_budget(); ++axis) {
                        for (const int sign : {-1, 1}) {
                            if (!within_budget()) break;
                            const auto saved = placed_[i];
                            const auto moments = cv::moments(saved.placement.points);
                            const cv::Point2f center(moments.m10 / moments.m00, moments.m01 / moments.m00);
                            double radius = 1.0;
                            for (const auto p : saved.placement.points) radius = std::max(radius, cv::norm(p - center));
                            const double angle = axis == 2 ? sign * step / radius : 0.0;
                            const cv::Point2f shift(axis == 0 ? sign * step : 0.0, axis == 1 ? sign * step : 0.0);
                            auto& candidate = placed_[i];
                            candidate.placement.rotation_radians = std::remainder(saved.placement.rotation_radians + angle, 2 * CV_PI);
                            candidate.placement.translation = rotate(saved.placement.translation - center, angle) + center + shift;
                            for (auto& p : candidate.placement.points) p = rotate(p - center, angle) + center + shift;
                            for (auto& triangle : candidate.triangles)
                                for (auto& p : triangle) p = rotate(p - center, angle) + center + shift;
                            const double error = layout_error();
                            if (error + 1e-9 < best_error) { best_error = error; improved = true; }
                            else candidate = saved;
                        }
                    }
                }
                if (!targets_.empty()) {
                    for (int axis = 0; axis < 2 && within_budget(); ++axis) {
                        for (const int sign : {-1, 1}) {
                            auto target = target_;
                            if (axis == 0) target.width = std::clamp(static_cast<float>(target.width + sign * step), min_size_.width, max_size_.width);
                            else target.height = std::clamp(static_cast<float>(target.height + sign * step), min_size_.height, max_size_.height);
                            const auto saved = target_;
                            set_target(target);
                            const double error = layout_error();
                            if (error + 1e-9 < best_error) { best_error = error; improved = true; }
                            else set_target(saved);
                        }
                    }
                }
                if (!improved) break;
            }
        }
        // accept_complete 只有通过原始全部阈值才会覆盖 result_，失败时原结果仍有效。
        if (!accept_complete()) { placed_ = original; set_target(original_target); }
    }

    // Try exchanging similarly sized pieces, with independent rigid half-turns.
    // Geometry remains authoritative: every trial is re-fitted and fully checked.
    void improve_exchanges() {
        const std::size_t count=result_.placements.size();
        if (count<2 || count>8) return;
        struct Trial { std::vector<Candidate> pieces; double score; bool accepted; };
        const auto within_exchange_budget=[&] {
            return std::chrono::duration<double>(std::chrono::steady_clock::now()-start_).count() <
                0.9*options_.time_limit_seconds;
        };
        for (int pass=0;pass<2;++pass) {
            const auto before=solution_revision_;
            const auto base=result_.placements;
            const auto target=result_.target;
            std::vector<Trial> trials;
            for (std::size_t i=0;i<count;++i) {
                for (std::size_t j=i+1;j<count;++j) {
                    const auto bounds_i=cv::boundingRect(base[i].points),bounds_j=cv::boundingRect(base[j].points);
                    const double margin=4*options_.position_tolerance+1;
                    if (std::abs(bounds_i.width-bounds_j.width)>margin ||
                        std::abs(bounds_i.height-bounds_j.height)>margin) continue;
                    for (int turns=0;turns<4;++turns) {
                        if (limit_reached() || !within_exchange_budget()) return;
                        Trial trial;
                        for (std::size_t k=0;k<count;++k) {
                            Candidate candidate;
                            candidate.placement=base[k];
                            if (k==i || k==j) {
                                auto& placement=candidate.placement;
                                const auto origin=base[k].center_gravity;
                                const auto destination=base[k==i ? j : i].center_gravity;
                                const double angle=(turns&(k==i ? 1 : 2)) ? CV_PI : 0;
                                placement.center_gravity=destination;
                                placement.rotation_radians=std::remainder(placement.rotation_radians+angle,2*CV_PI);
                                placement.translation=rotate(placement.translation-origin,angle)+destination;
                                for (auto& p : placement.points) p=rotate(p-origin,angle)+destination;
                            }
                            candidate.triangles=triangulate(candidate.placement.points);
                            trial.pieces.push_back(std::move(candidate));
                        }
                        placed_=trial.pieces;
                        set_target(target);
                        const auto appearance=layout_score(placed_);
                        trial.accepted=appearance.accepted;
                        trial.score=appearance_score(placed_)+boundary_score(placed_)+appearance.score-10*layout_error();
                        trials.push_back(std::move(trial));
                    }
                }
            }
            std::stable_sort(trials.begin(),trials.end(),[](const Trial& a,const Trial& b) {
                return a.accepted!=b.accepted ? a.accepted : a.score>b.score;
            });
            for (const auto& trial : trials) {
                if (limit_reached() || !within_exchange_budget()) return;
                placed_=trial.pieces;
                set_target(target);
                ++result_.explored_nodes;
                const auto previous=solution_revision_;
                if (!accept_complete()) improve_solution();
                if (solution_revision_>previous) break;
            }
            if (solution_revision_==before) break;
        }
    }

    // Similar card fragments can form a valid rectangle with a whole strip
    // reversed. Test rigid half-turns of rectangular subassemblies and keep the
    // original complete geometry checks for every candidate.
    void improve_appearance() {
        const std::size_t count=result_.placements.size();
        if (count<2 || count>8) return;
        for (int pass=0;pass<2;++pass) {
            const auto before=solution_revision_;
            const auto base=result_.placements;
            const auto target=result_.target;
            for (std::size_t subset=1;subset+1<(std::size_t{1}<<count);++subset) {
                if (limit_reached()) return;
                std::vector<cv::Point2f> group;
                double area=0;
                for (std::size_t i=0;i<count;++i) {
                    if (!(subset&(std::size_t{1}<<i))) continue;
                    group.insert(group.end(),base[i].points.begin(),base[i].points.end());
                    area+=std::abs(cv::contourArea(base[i].points));
                }
                float left=group.front().x,right=left,top=group.front().y,bottom=top;
                for (const auto p : group) {
                    left=std::min(left,p.x);right=std::max(right,p.x);
                    top=std::min(top,p.y);bottom=std::max(bottom,p.y);
                }
                const double rectangularity=options_.min_coverage-options_.max_overlap_ratio-options_.max_outside_ratio;
                if (area<(right-left)*(bottom-top)*rectangularity) continue;
                const cv::Point2f twice_center(left+right,top+bottom);
                placed_.clear();
                for (std::size_t i=0;i<count;++i) {
                    Candidate candidate;
                    candidate.placement=base[i];
                    if (subset&(std::size_t{1}<<i)) {
                        auto& placement=candidate.placement;
                        placement.rotation_radians=std::remainder(placement.rotation_radians+CV_PI,2*CV_PI);
                        placement.translation=twice_center-placement.translation;
                        for (auto& point : placement.points) point=twice_center-point;
                    }
                    candidate.triangles=triangulate(candidate.placement.points);
                    placed_.push_back(std::move(candidate));
                }
                set_target(target);
                ++result_.explored_nodes;
                if (!accept_complete()) improve_solution();
            }
            if (solution_revision_==before) break;
        }
    }

    bool visit() {
        if (limit_reached()) return false;
        ++result_.explored_nodes;
        if (placed_.size() == shapes_.size()) {
            if (!accept_complete()) return false;
            improve_solution();
            return true;
        }
        // 大块优先只是搜索顺序。失败会尝试其他位置和其他拼块，不固定第一块的角落。
        for (std::size_t i = 0; i < shapes_.size(); ++i) {
            if (used_[i]) continue;
            auto choices = candidates(shapes_[i]);
            if (result_.search_limit_reached) return false;
            used_[i] = true;
            for (auto& choice : choices) {
                placed_.push_back(std::move(choice));
                if (visit()) return true;
                placed_.pop_back();
                if (result_.search_limit_reached) break;
            }
            used_[i] = false;
            if (result_.search_limit_reached) return false;
        }
        return false;
    }

    cv::Rect2f target_;
    const PuzzleOptions& options_;
    std::chrono::steady_clock::time_point start_;
    Polygon border_;
    std::vector<Triangle> border_triangles_;
    std::vector<Shape> shapes_;
    std::vector<bool> used_;
    std::vector<Candidate> placed_;
    PuzzleResult result_;
    std::size_t solution_revision_=0;
    double best_solution_quality_ = -std::numeric_limits<double>::infinity();
    bool valid_ = true;
    std::vector<cv::Rect2f> targets_;
    cv::Size2f min_size_, max_size_;
};
} // namespace

PuzzleResult solve_puzzle(const std::vector<std::vector<cv::Point2f>>& polygons,
                          const cv::Rect2f& target, const PuzzleOptions& options) {
    if (!std::isfinite(target.x) || !std::isfinite(target.y) ||
        !std::isfinite(target.width) || !std::isfinite(target.height) ||
        target.width <= 0 || target.height <= 0 || !valid_options(options)) return {};
    return Search(polygons, target, options).run();
}

PuzzleResult solve_puzzle_in_range(const std::vector<std::vector<cv::Point2f>>& polygons,
                                   cv::Point2f origin, cv::Size2f min_size, cv::Size2f max_size,
                                   const PuzzleOptions& options) {
    if (!valid_options(options) || !std::isfinite(origin.x) || !std::isfinite(origin.y) ||
        !std::isfinite(min_size.width) || !std::isfinite(min_size.height) ||
        !std::isfinite(max_size.width) || !std::isfinite(max_size.height) ||
        min_size.width <= 0 || min_size.height <= 0 ||
        max_size.width < min_size.width || max_size.height < min_size.height) return {};
    double area = 0.0;
    std::vector<double> edges;
    std::map<int, double> sums{{0, 0.0}};
    for (const auto& polygon : polygons) {
        if (polygon.empty()) continue;
        if (polygon.size() < 3) return {};
        for (const auto p : polygon) if (!std::isfinite(p.x) || !std::isfinite(p.y)) return {};
        area += std::abs(cv::contourArea(polygon));
        std::vector<double> piece_edges;
        for (std::size_t i = 0; i < polygon.size(); ++i) {
            const double length = cv::norm(polygon[(i + 1) % polygon.size()] - polygon[i]);
            edges.push_back(length);
            piece_edges.push_back(length);
        }
        auto next = sums;
        for (const auto& item : sums)
            for (const auto length : piece_edges) {
                const double sum = item.second + length;
                if (sum <= std::max(max_size.width, max_size.height) + 2 * options.position_tolerance)
                    next.emplace(static_cast<int>(std::lround(sum * 2)), sum);
            }
        sums = std::move(next);
    }
    if (area <= 0 || edges.empty()) return {};
    const auto feasible_area = [&](double w, double h) {
        return area >= w * h * options.min_coverage &&
            area <= w * h * (1.0 + options.max_overlap_ratio + options.max_outside_ratio);
    };
    if (area < min_size.area() * options.min_coverage ||
        area > max_size.area() * (1.0 + options.max_overlap_ratio + options.max_outside_ratio)) return {};
    std::vector<double> widths{min_size.width, max_size.width}, heights{min_size.height, max_size.height};
    for (const auto& item : sums) {
        const double value = item.second, tolerance = 2 * options.position_tolerance;
        if (value >= min_size.width - tolerance && value <= max_size.width + tolerance)
            widths.push_back(std::clamp(value, static_cast<double>(min_size.width), static_cast<double>(max_size.width)));
        if (value >= min_size.height - tolerance && value <= max_size.height + tolerance)
            heights.push_back(std::clamp(value, static_cast<double>(min_size.height), static_cast<double>(max_size.height)));
    }
    struct SizeCandidate { cv::Rect2f target; double error; };
    std::vector<SizeCandidate> candidates;
    const auto support = [&](double length) {
        double distance = length;
        for (const auto edge : edges) distance = std::min(distance, std::abs(edge - length));
        return std::max(0.0, 1.0 - distance / std::max(1e-3, options.position_tolerance));
    };
    for (const auto w : widths)
        for (const auto h : heights) {
            if (!feasible_area(w, h)) continue;
            const double error = std::abs(area - w * h) / area - 0.02 * (support(w) + support(h));
            candidates.push_back({cv::Rect2f(origin.x, origin.y, w, h), error});
        }
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.error < b.error; });
    std::vector<cv::Rect2f> targets;
    const double separation = std::max(1.0, 2 * options.position_tolerance);
    const auto add = [&](cv::Rect2f target) {
        if (targets.size() >= 16 || !feasible_area(target.width, target.height)) return;
        for (const auto old : targets)
            if (std::abs(old.width - target.width) < separation && std::abs(old.height - target.height) < separation) return;
        targets.push_back(target);
    };
    // 首先保留数据支持最强的尺寸，再用面积关系覆盖剩余长宽比，避免仅按面积猜一个尺寸。
    for (const auto& candidate : candidates) { add(candidate.target); if (targets.size() == 4) break; }
    for (const int index : {0, 12, 6, 3, 9, 1, 11, 5, 7, 2, 10, 4, 8}) {
        const double w = min_size.width + (max_size.width - min_size.width) * index / 12.0;
        const double h = std::clamp(area / w, static_cast<double>(min_size.height), static_cast<double>(max_size.height));
        add(cv::Rect2f(origin.x, origin.y, w, h));
    }
    for (const auto& candidate : candidates) add(candidate.target);
    if (targets.empty()) return {};
    return Search(polygons, targets.front(), options, targets, min_size, max_size).run();
}
