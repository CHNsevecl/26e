#include "card_puzzle.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>

namespace {
cv::Point2f rotate(cv::Point2f p, double a) {
    return {static_cast<float>(std::cos(a)*p.x-std::sin(a)*p.y),
            static_cast<float>(std::sin(a)*p.x+std::cos(a)*p.y)};
}

cv::Mat estimate_background(const cv::Mat& image, const cv::Mat& gray, double seed_limit, double fallback) {
    std::vector<cv::Vec<double,6>> positions;
    std::vector<cv::Vec3d> colors;
    for (int y=0;y<image.rows;y+=8)
        for (int x=0;x<image.cols;x+=8) {
            if (gray.at<uchar>(y,x)>seed_limit) continue;
            const double u=static_cast<double>(x)/image.cols,v=static_cast<double>(y)/image.rows;
            positions.push_back({1,u,v,u*u,u*v,v*v});
            colors.emplace_back(image.at<cv::Vec3b>(y,x));
        }
    cv::Mat coefficients=cv::Mat::zeros(6,3,CV_64F);
    coefficients.row(0).setTo(fallback);
    std::vector<bool> selected(positions.size(),true);
    for (int pass=0;pass<5 && positions.size()>=24;++pass) {
        const int count=static_cast<int>(std::count(selected.begin(),selected.end(),true));
        if (count<24) break;
        cv::Mat design(count,6,CV_64F),values(count,3,CV_64F);
        int row=0;
        for (std::size_t i=0;i<positions.size();++i) {
            if (!selected[i]) continue;
            for (int k=0;k<6;++k) design.at<double>(row,k)=positions[i][k];
            for (int k=0;k<3;++k) values.at<double>(row,k)=colors[i][k];
            ++row;
        }
        cv::Mat fitted;
        if (!cv::solve(design,values,fitted,cv::DECOMP_SVD) || !cv::checkRange(fitted)) break;
        coefficients=fitted;
        std::vector<double> errors;
        for (std::size_t i=0;i<positions.size();++i) {
            cv::Vec3d predicted;
            for (int k=0;k<6;++k)
                for (int channel=0;channel<3;++channel)
                    predicted[channel]+=positions[i][k]*coefficients.at<double>(k,channel);
            errors.push_back(cv::norm(colors[i]-predicted));
        }
        auto ordered=errors;
        std::nth_element(ordered.begin(),ordered.begin()+ordered.size()*4/5,ordered.end());
        const double threshold=std::max(8.0,ordered[ordered.size()*4/5]);
        for (std::size_t i=0;i<selected.size();++i) selected[i]=errors[i]<=threshold;
    }
    cv::Mat background(image.size(),CV_8UC3);
    for (int y=0;y<image.rows;++y)
        for (int x=0;x<image.cols;++x) {
            const double u=static_cast<double>(x)/image.cols,v=static_cast<double>(y)/image.rows;
            const cv::Vec<double,6> basis(1,u,v,u*u,u*v,v*v);
            cv::Vec3d color;
            for (int k=0;k<6;++k)
                for (int channel=0;channel<3;++channel)
                    color[channel]+=basis[k]*coefficients.at<double>(k,channel);
            background.at<cv::Vec3b>(y,x)=color;
        }
    return background;
}

// 从实体轮廓的长直边拟合几何模型；短圆弧不作为新的切割边。
// 原始掩膜仍保留圆角，几何模型只供求解器使用。
std::vector<cv::Point2f> fit_piece_edges(const std::vector<cv::Point>& contour,bool recover_short_cuts=false) {
    const double perimeter=cv::arcLength(contour,true);
    std::vector<cv::Point> seed;
    cv::approxPolyDP(contour,seed,perimeter*0.01,true);
    // A rounded/noisy end can split one long straight side into two seed edges.
    // Merge only a shallow convex turn supported by the original contour pixels;
    // sharp bevels and actual concave cuts must remain separate edges.
    bool merged=true;
    while (merged && seed.size()>3) {
        merged=false;
        for (std::size_t i=0;i<seed.size();++i) {
            const auto a=seed[(i+seed.size()-1)%seed.size()],b=seed[i],c=seed[(i+1)%seed.size()];
            const cv::Point2f before=b-a,after=c-b;
            const double lengths=cv::norm(before)*cv::norm(after);
            const double turn=before.x*after.y-before.y*after.x;
            if (lengths<1 || before.dot(after)<std::cos(12*CV_PI/180)*lengths ||
                turn*cv::contourArea(seed,true)<=0) continue;
            const auto first=std::find(contour.begin(),contour.end(),a);
            const auto last=std::find(contour.begin(),contour.end(),c);
            if (first==contour.end() || last==contour.end()) continue;
            std::vector<cv::Point2f> arc;
            for (std::size_t j=first-contour.begin();;j=(j+1)%contour.size()) {
                arc.push_back(contour[j]);
                if (j==static_cast<std::size_t>(last-contour.begin())) break;
            }
            if (arc.size()<20) continue;
            cv::Vec4f line;
            cv::fitLine(arc,line,cv::DIST_L2,0,0.01,0.01);
            std::vector<double> errors;
            for (const auto& point : arc)
                errors.push_back(std::abs((point.x-line[2])*line[1]-(point.y-line[3])*line[0]));
            std::sort(errors.begin(),errors.end());
            if (errors[errors.size()*9/10]>1.5 || errors[errors.size()*19/20]>3.0) continue;
            seed.erase(seed.begin()+i);
            merged=true;
            break;
        }
    }
    const std::vector<cv::Point2f> fallback(seed.begin(),seed.end());
    if (seed.size()<3) return fallback;
    // Near-right-angle seed corners can lie inside a rounded physical edge.
    // Fit the supported sides before deciding to retain an already-straight
    // polygon; a tiny change in angle must not bypass rounded-corner recovery.
    std::vector<std::size_t> indices;
    for (const auto& corner : seed) {
        const auto found=std::find(contour.begin(),contour.end(),corner);
        if (found==contour.end()) return fallback;
        indices.push_back(found-contour.begin());
    }
    std::vector<cv::Vec4f> lines;
    std::vector<std::vector<cv::Point2f>> line_supports;
    std::vector<std::pair<std::size_t,std::size_t>> line_arcs;
    bool all_straight=true;
    for (std::size_t k=0;k<seed.size();++k) {
        const cv::Point2f a=seed[k],b=seed[(k+1)%seed.size()];
        const double length=cv::norm(b-a);
        if (length<1) return fallback;
        const auto direction=(b-a)/static_cast<float>(length);
        std::vector<cv::Point2f> arc,support;
        std::vector<double> deviations;
        for (std::size_t j=indices[k];;j=(j+1)%contour.size()) {
            const cv::Point2f p=contour[j],delta=p-a;
            arc.push_back(p);
            deviations.push_back(std::abs(delta.x*direction.y-delta.y*direction.x));
            const double projection=delta.dot(direction);
            if (projection>0.15*length && projection<0.85*length) support.push_back(p);
            if (j==indices[(k+1)%seed.size()]) break;
        }
        std::sort(deviations.begin(),deviations.end());
        const double curvature=deviations[deviations.size()*4/5];
        if (curvature>1.2) all_straight=false;
        // 真正的短直边仍保留。仅合并长度较短、明显弯曲的圆角过渡。
        const double corner_curvature=std::clamp(0.05*length,0.65,1.2);
        if (!recover_short_cuts && length<0.08*perimeter && curvature>corner_curvature) continue;

        if (support.size()<8) support=arc;
        if (support.size()<2) return fallback;
        cv::Vec4f line;
        cv::fitLine(support,line,cv::DIST_HUBER,0,0.01,0.01);
        for (int pass=0;pass<3;++pass) {
            std::vector<double> errors;
            for (const auto& p : support)
                errors.push_back(std::abs((p.x-line[2])*line[1]-(p.y-line[3])*line[0]));
            auto ordered=errors;
            std::sort(ordered.begin(),ordered.end());
            const double threshold=std::max(1.2,ordered[ordered.size()*4/5]);
            std::vector<cv::Point2f> core;
            for (std::size_t j=0;j<support.size();++j)
                if (errors[j]<=threshold) core.push_back(support[j]);
            if (core.size()>=6) cv::fitLine(core,line,cv::DIST_L2,0,0.01,0.01);
        }
        const bool aligned=std::abs(line[0]*direction.x+line[1]*direction.y)>=std::cos(12*CV_PI/180);
        if (recover_short_cuts && length<0.08*perimeter) {
            std::vector<double> arc_errors;
            for (const auto point : arc)
                arc_errors.push_back(std::abs((point.x-line[2])*line[1]-(point.y-line[3])*line[0]));
            std::sort(arc_errors.begin(),arc_errors.end());
            // Rounded endpoints can bend the seed chord of a real straight cut.
            // Use its actual line support before discarding the entire segment;
            // unstable short flower arcs must not abort all the long-edge fits.
            if (!aligned || (curvature>corner_curvature &&
                (length<8*scale || arc_errors[arc_errors.size()*4/5]>corner_curvature))) continue;
        } else if (!aligned) return fallback;
        lines.push_back(line);
        line_supports.push_back(std::move(support));
        line_arcs.emplace_back(k,k);
    }
    if (lines.size()==seed.size() && all_straight) return fallback;
    // Discarded curved printing can leave two parts of the same straight cut.
    // Merge only when both sets of actual contour samples support one line;
    // offset parallel edges and straight concave cuts must stay separate.
    bool joined=true;
    while (joined && lines.size()>3) {
        joined=false;
        for (std::size_t i=0;i<lines.size();++i) {
            const auto previous=(i+lines.size()-1)%lines.size();
            const auto& a=lines[previous];
            const auto& b=lines[i];
            if (std::abs(a[0]*b[1]-a[1]*b[0])>std::sin(3*CV_PI/180)) continue;
            auto support=line_supports[previous];
            support.insert(support.end(),line_supports[i].begin(),line_supports[i].end());
            cv::Vec4f common;
            cv::fitLine(support,common,cv::DIST_L2,0,0.01,0.01);
            std::vector<double> errors;
            for (const auto point : support)
                errors.push_back(std::abs((point.x-common[2])*common[1]-(point.y-common[3])*common[0]));
            std::sort(errors.begin(),errors.end());
            if (errors[errors.size()*9/10]>1.5 || errors[errors.size()*19/20]>3) continue;
            // This repair is for an inward curved print notch. Outward spurs
            // retain the existing geometry-only opening fallback instead.
            const auto moments=cv::moments(contour);
            const cv::Point2f center(moments.m10/moments.m00,moments.m01/moments.m00);
            const double toward_center=(center.x-common[2])*common[1]-(center.y-common[3])*common[0];
            if (std::abs(toward_center)<1e-3) continue;
            double deepest=0,outward=0;
            for (std::size_t j=(line_arcs[previous].second+1)%seed.size();;j=(j+1)%seed.size()) {
                const auto point=seed[j];
                const double depth=((point.x-common[2])*common[1]-(point.y-common[3])*common[0])*
                                   (toward_center>0 ? 1 : -1);
                deepest=std::max(deepest,depth);outward=std::min(outward,depth);
                if (j==line_arcs[i].first) break;
            }
            if (deepest<2*scale || outward<-2) continue;
            line_arcs[previous].second=line_arcs[i].second;
            lines[previous]=common;
            line_supports[previous]=std::move(support);
            lines.erase(lines.begin()+i);
            line_supports.erase(line_supports.begin()+i);
            line_arcs.erase(line_arcs.begin()+i);
            joined=true;
            break;
        }
    }
    if (lines.size()<3) return fallback;
    std::vector<cv::Point2f> polygon;
    for (std::size_t i=0;i<lines.size();++i) {
        const auto& previous=lines[(i+lines.size()-1)%lines.size()];
        const auto& current=lines[i];
        const double determinant=previous[0]*current[1]-previous[1]*current[0];
        if (std::abs(determinant)<0.05) return fallback;
        const cv::Point2f difference(current[2]-previous[2],current[3]-previous[3]);
        const double t=(difference.x*current[1]-difference.y*current[0])/determinant;
        const cv::Point2f corner(previous[2]+t*previous[0],previous[3]+t*previous[1]);
        if (!std::isfinite(corner.x) || !std::isfinite(corner.y) ||
            std::abs(cv::pointPolygonTest(contour,corner,true))>std::min(8.0*scale,perimeter*0.04)) return fallback;
        polygon.push_back(corner);
    }
    const double physical_area=std::abs(cv::contourArea(contour));
    const double model_area=std::abs(cv::contourArea(polygon));
    // A contour through pixel centers has a half-pixel boundary uncertainty.
    // Apply that area allowance only to the supported-cut recovery fallback;
    // stable models and the final puzzle coverage/overlap checks stay unchanged.
    const double boundary_area=recover_short_cuts ? perimeter*0.5 : 0;
    if (model_area<0.95*physical_area || model_area>1.05*physical_area+boundary_area ||
        cv::contourArea(polygon,true)*cv::contourArea(contour,true)<=0) return fallback;
    return polygon;
}

// Only the geometry estimate is cleaned. Keep the physical contour, texture,
// masks and pickup centroid unchanged when a small attached spur adds vertices.
std::vector<cv::Point2f> fit_piece_model(const std::vector<cv::Point>& contour) {
    auto best=fit_piece_edges(contour);
    if (best.size()<=4 || cv::isContourConvex(best)) return best;
    const auto bounds=cv::boundingRect(contour);
    const int padding=4;
    cv::Mat mask=cv::Mat::zeros(bounds.height+2*padding,bounds.width+2*padding,CV_8UC1);
    std::vector<cv::Point> local;
    for (const auto point : contour) local.push_back(point-bounds.tl()+cv::Point(padding,padding));
    cv::drawContours(mask,std::vector<std::vector<cv::Point>>{local},0,cv::Scalar(255),cv::FILLED);
    const double area=std::abs(cv::contourArea(contour));
    for (const int diameter : {3,5,7}) {
        cv::Mat cleaned;
        cv::morphologyEx(mask,cleaned,cv::MORPH_OPEN,
            cv::getStructuringElement(cv::MORPH_ELLIPSE,{diameter,diameter}));
        std::vector<std::vector<cv::Point>> components;
        cv::findContours(cleaned,components,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
        if (components.size()!=1 || std::abs(cv::contourArea(components.front()))<0.98*area) continue;
        for (auto& point : components.front()) point+=bounds.tl()-cv::Point(padding,padding);
        auto model=fit_piece_edges(components.front());
        if (model.size()>=3 && model.size()<best.size()) best=std::move(model);
    }
    if (best.size()>4 && !cv::isContourConvex(best)) {
        // Retain every stable existing model. Retry only unresolved notched
        // contours with straight-support recovery for short seed segments.
        auto recovered=fit_piece_edges(contour,true);
        if (recovered.size()>=3 && recovered.size()<best.size()) best=std::move(recovered);
    }
    return best;
}

// 黑色图案恰好碰到切边时会表现成缺口。仅在缺口比边外的黑纸明显更黑时补回；
// 真正的凹形切口与背景同色，保留原形状，不直接取整个凸包。
std::vector<cv::Point> repair_ink_notches(const cv::Mat& gray, const std::vector<cv::Point>& contour) {
    if (contour.size()<4 || cv::isContourConvex(contour)) return contour;
    std::vector<int> hull;
    cv::convexHull(contour,hull,false,false);
    if (hull.size()<3) return contour;
    std::vector<cv::Vec4i> defects;
    cv::convexityDefects(contour,hull,defects);
    cv::Mat filled=cv::Mat::zeros(gray.size(),CV_8UC1);
    cv::drawContours(filled,std::vector<std::vector<cv::Point>>{contour},0,cv::Scalar(255),cv::FILLED);
    const double area=std::abs(cv::contourArea(contour));
    bool changed=false;
    std::vector<cv::Point> hull_points;
    cv::convexHull(contour,hull_points);
    for (const auto& defect : defects) {
        const cv::Point2f hull_a=contour[defect[0]],hull_b=contour[defect[1]];
        const double hull_length=cv::norm(hull_b-hull_a);
        if (hull_length<3 || defect[3]/256.0<2) continue;
        std::vector<cv::Point> arc;
        for (std::size_t index=defect[0];;index=(index+1)%contour.size()) {
            arc.push_back(contour[index]);
            if (index==static_cast<std::size_t>(defect[1])) break;
        }
        std::size_t first=arc.size(),last=0;
        for (std::size_t k=0;k<arc.size();++k) {
            const auto v=cv::Point2f(arc[k])-hull_a,w=hull_b-hull_a;
            if (std::abs(v.x*w.y-v.y*w.x)/hull_length>0.75) {
                first=std::min(first,k);
                last=k;
            }
        }
        if (first==arc.size()) continue;
        first=first>0 ? first-1 : first;
        last=std::min(last+1,arc.size()-1);
        const std::vector<cv::Point> notch(arc.begin()+first,arc.begin()+last+1);
        const cv::Point2f a=notch.front(),b=notch.back();
        const double length=cv::norm(b-a);
        if (length<3 || length>16*scale) continue;
        const auto direction=(b-a)/static_cast<float>(length);
        cv::Point2f outward(-direction.y,direction.x);
        const auto middle=(a+b)*0.5f;
        if (cv::pointPolygonTest(hull_points,middle+outward*3.0f,false)>=0) outward=-outward;
        std::vector<uchar> background;
        for (int k=1;k<10;++k) {
            const auto p=a+(b-a)*(k/10.0f)+outward*5.0f;
            const cv::Point pixel(cvRound(p.x),cvRound(p.y));
            if (pixel.x>=0 && pixel.y>=0 && pixel.x<gray.cols && pixel.y<gray.rows)
                background.push_back(gray.at<uchar>(pixel));
        }
        if (background.size()<5) continue;
        std::sort(background.begin(),background.end());
        const int black_paper=background[background.size()/2];
        if (black_paper<15) continue; // 油墨和背景无可辨差异时不能猜补。
        if (notch.size()<3) continue;
        const auto box=cv::boundingRect(notch);
        std::vector<cv::Point> local;
        for (const auto& p : notch) local.push_back(p-box.tl());
        cv::Mat patch=cv::Mat::zeros(box.size(),CV_8UC1);
        cv::drawContours(patch,std::vector<std::vector<cv::Point>>{local},0,cv::Scalar(255),cv::FILLED);
        int missing=0,ink=0;
        for (int y=0;y<patch.rows;++y)
            for (int x=0;x<patch.cols;++x)
                if (patch.at<uchar>(y,x) && !filled.at<uchar>(y+box.y,x+box.x)) {
                    ++missing;
                    ink+=gray.at<uchar>(y+box.y,x+box.x)<black_paper-12;
                }
        if (missing<12 || missing>area*0.15 || ink<missing*0.70) continue;
        cv::bitwise_or(filled(box),patch,filled(box));
        changed=true;
    }
    if (!changed) return contour;
    std::vector<std::vector<cv::Point>> repaired;
    cv::findContours(filled,repaired,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
    if (repaired.empty()) return contour;
    return *std::max_element(repaired.begin(),repaired.end(),[](const auto& a,const auto& b) {
        return cv::contourArea(a)<cv::contourArea(b);
    });
}

bool sample(const CardRecognition& recognition, const PuzzlePlacement& placement,
            cv::Point2f point, cv::Vec3d& color) {
    if (placement.source_index >= recognition.fragments.size()) return false;
    const auto& fragment = recognition.fragments[placement.source_index];
    const auto original = rotate(point-placement.translation,-placement.rotation_radians);
    const cv::Point local(cvRound(original.x)-fragment.bounds.x,cvRound(original.y)-fragment.bounds.y);
    if (local.x < 0 || local.y < 0 || local.x >= fragment.mask.cols || local.y >= fragment.mask.rows ||
        !(fragment.texture_mask.empty() ? fragment.mask : fragment.texture_mask).at<uchar>(local)) return false;
    color = fragment.image.at<cv::Vec3b>(local);
    return true;
}

// 普通扑克牌的白色外边应位于牌面外框，避免相似四边形把两条白边拼到牌面中间。
std::vector<std::vector<double>> border_evidence(const CardRecognition& recognition) {
    std::vector<std::vector<double>> evidence;
    for (std::size_t id=0; id<recognition.fragments.size(); ++id) {
        const auto& fragment=recognition.fragments[id];
        std::vector<double> levels;
        for (int y=0; y<fragment.image.rows; y+=3)
            for (int x=0; x<fragment.image.cols; x+=3)
                if (fragment.mask.at<uchar>(y,x)) {
                    const auto c=fragment.image.at<cv::Vec3b>(y,x);
                    levels.push_back(std::min({c[0],c[1],c[2]}));
                }
        // 每块独立估计纸白，减小照明不均导致的白边评分差异。
        std::sort(levels.begin(),levels.end());
        const double white=levels.empty() ? 255 : std::max(80.0,levels[levels.size()*9/10]);
        const auto& points=recognition.pieces.pieces_point[id];
        PuzzlePlacement identity;
        identity.source_index=id;
        const double sign=cv::contourArea(points,true)>0 ? 1 : -1;
        std::vector<double> edges;
        for (std::size_t j=0; j<points.size(); ++j) {
            const auto p=points[j],v=points[(j+1)%points.size()]-p;
            const double length=cv::norm(v);
            if (length<1e-6) { edges.push_back(0); continue; }
            const cv::Point2f inward(-v.y*sign/length,v.x*sign/length);
            double ink=0;
            int count=0;
            // 检查边内多个深度，不能把裁边处的一排浅色像素当作整条白框。
            for (int k=0;k<48;++k)
                for (const float depth : {3.f,7.f,12.f}) {
                    cv::Vec3d color;
                    if (!sample(recognition,identity,p+v*((k+0.5f)/48)+inward*depth,color)) continue;
                    ink+=std::clamp(1-std::min({color[0],color[1],color[2]})/white,0.0,1.0);
                    ++count;
                }
            // 软评分：无足够实体像素时不作判断；不改变几何验收阈值。
            edges.push_back(count>=72 ? 0.2-2.0*ink/count : 0);
        }
        evidence.push_back(edges);
    }
    return evidence;
}

// A physical rounded corner removes paper near the intersection of two fitted
// straight edges. Blank card interiors cannot provide this perimeter evidence.
std::vector<std::vector<double>> corner_evidence(const CardRecognition& recognition) {
    std::vector<std::vector<double>> evidence;
    const double radius=3.0*scale;
    for (std::size_t id=0;id<recognition.fragments.size();++id) {
        const auto& fragment=recognition.fragments[id];
        const auto& polygon=recognition.pieces.pieces_point[id];
        std::vector<double> missing;
        for (std::size_t i=0;i<polygon.size();++i) {
            const auto vertex=polygon[i];
            const cv::Point2f incoming=vertex-polygon[(i+polygon.size()-1)%polygon.size()];
            const cv::Point2f outgoing=polygon[(i+1)%polygon.size()]-vertex;
            const double lengths=cv::norm(incoming)*cv::norm(outgoing);
            const double turn=incoming.x*outgoing.y-incoming.y*outgoing.x;
            // Original card corners are convex and approximately perpendicular.
            // Blur at a sharp acute cut must not count as a rounded outer corner.
            if (lengths<1 || turn*cv::contourArea(polygon,true)<=0 ||
                std::abs(incoming.dot(outgoing))>std::sin(15*CV_PI/180)*lengths) {
                missing.push_back(0);
                continue;
            }
            int inside=0,absent=0;
            for (int y=static_cast<int>(vertex.y-radius);y<=vertex.y+radius;++y)
                for (int x=static_cast<int>(vertex.x-radius);x<=vertex.x+radius;++x) {
                    const cv::Point2f point(x+0.5f,y+0.5f);
                    if (cv::norm(point-vertex)>radius || cv::pointPolygonTest(polygon,point,false)<0) continue;
                    ++inside;
                    const auto local=cv::Point(x,y)-fragment.bounds.tl();
                    if (local.x<0 || local.y<0 || local.x>=fragment.mask.cols || local.y>=fragment.mask.rows ||
                        !fragment.mask.at<uchar>(local)) ++absent;
                }
            missing.push_back(inside ? static_cast<double>(absent)/inside : 0);
        }
        auto ordered=missing;
        std::sort(ordered.begin(),ordered.end());
        const double median=(ordered[ordered.size()/2]+ordered[(ordered.size()-1)/2])*0.5;
        for (auto& value : missing) value=std::max(0.0,value-median-0.08);
        evidence.push_back(std::move(missing));
    }
    return evidence;
}

cv::Mat render(const CardRecognition& recognition, const std::vector<PuzzlePlacement>& placements) {
    cv::Mat canvas = cv::Mat::zeros(papers_height*scale,papers_width*scale,CV_8UC3);
    for (const auto& placement : placements) {
        const auto& fragment = recognition.fragments.at(placement.source_index);
        const double c=std::cos(placement.rotation_radians),s=std::sin(placement.rotation_radians);
        const cv::Matx23d transform(c,-s,c*fragment.bounds.x-s*fragment.bounds.y+placement.translation.x,
                                   s,c,s*fragment.bounds.x+c*fragment.bounds.y+placement.translation.y);
        cv::Mat image,mask;
        cv::warpAffine(fragment.image,image,transform,canvas.size(),cv::INTER_LINEAR);
        cv::warpAffine(fragment.texture_mask.empty() ? fragment.mask : fragment.texture_mask,
                       mask,transform,canvas.size(),cv::INTER_NEAREST);
        image.copyTo(canvas,mask);
    }
    return canvas;
}
}

CardRecognition recognize_card_pieces(const cv::Mat& warped, const CardPuzzleOptions& options) {
    CardRecognition recognition;
    if (warped.empty() || warped.type()!=CV_8UC3 || warped.rows<16 || warped.cols<16) {
        recognition.error="扑克牌识别需要非空 BGR 透视图";
        return recognition;
    }
    if (!std::isfinite(options.min_piece_area_mm2) || options.min_piece_area_mm2<=0 ||
        !std::isfinite(options.closing_mm) || options.closing_mm<0 || options.closing_mm>5) {
        recognition.error="牌片面积或闭运算参数无效";
        return recognition;
    }
    cv::Mat gray,blurred,hsv,raw;
    cv::cvtColor(warped,gray,cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray,blurred,{3,3},0.8);
    cv::cvtColor(warped,hsv,cv::COLOR_BGR2HSV);
    std::vector<uchar> levels;
    for (int y=0;y<gray.rows;y+=4)
        for (int x=0;x<gray.cols;x+=4) levels.push_back(gray.at<uchar>(y,x));
    std::sort(levels.begin(),levels.end());
    const double background=levels[levels.size()/4];
    const double otsu=cv::threshold(blurred,raw,0,255,cv::THRESH_BINARY|cv::THRESH_OTSU);
    recognition.background_image=estimate_background(warped,gray,std::max(otsu*0.85,background+8),background);
    for (int y=0;y<raw.rows;++y)
        for (int x=0;x<raw.cols;++x) {
            const auto paper=recognition.background_image.at<cv::Vec3b>(y,x);
            const double local=0.114*paper[0]+0.587*paper[1]+0.299*paper[2];
            const double cutoff=local+std::max(18.0,0.45*std::max(0.0,otsu-local));
            raw.at<uchar>(y,x)=blurred.at<uchar>(y,x)>cutoff ? 255 : 0;
            const auto color=hsv.at<cv::Vec3b>(y,x);
            if (color[1]>70 && color[2]>std::max({paper[0],paper[1],paper[2]})+25) raw.at<uchar>(y,x)=255;
        }
    const int kernel=std::max(1,static_cast<int>(std::lround(options.closing_mm*scale))|1);
    if (kernel>1)
        cv::morphologyEx(raw,raw,cv::MORPH_CLOSE,cv::getStructuringElement(cv::MORPH_ELLIPSE,{kernel,kernel}));
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(raw,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
    const double min_area=options.min_piece_area_mm2*scale*scale;
    contours.erase(std::remove_if(contours.begin(),contours.end(),[&](const auto& contour) {
        const auto bounds=cv::boundingRect(contour);
        return cv::contourArea(contour)<min_area || bounds.x<=1 || bounds.y<=1 ||
            bounds.br().x>=warped.cols-1 || bounds.br().y>=warped.rows-1;
    }),contours.end());
    for (auto& contour : contours) contour=repair_ink_notches(gray,contour);
    std::sort(contours.begin(),contours.end(),[](const auto& a,const auto& b) {
        const auto ma=cv::moments(a),mb=cv::moments(b);
        const double ya=ma.m01/ma.m00,yb=mb.m01/mb.m00;
        return ya!=yb ? ya<yb : ma.m10/ma.m00<mb.m10/mb.m00;
    });
    recognition.foreground_mask=cv::Mat::zeros(warped.size(),CV_8UC1);
    recognition.debug_image=warped.clone();
    auto& pieces=recognition.pieces;
    for (const auto& contour : contours) {
        const auto polygon=fit_piece_model(contour);
        if (polygon.size()<3 || std::abs(cv::contourArea(polygon))<min_area) continue;
        const auto moments=cv::moments(contour);
        if (std::abs(moments.m00)<1e-6) continue;
        const std::size_t id=pieces.pieces_point.size();
        const cv::Point2f center(moments.m10/moments.m00,moments.m01/moments.m00);
        pieces.pieces_point.push_back(polygon);
        pieces.center_gravity.push_back(center);
        pieces.gra_p_vector.emplace_back();
        pieces.pieces_lines.emplace_back();
        pieces.turn_angle.push_back(0);
        pieces.pieces_lines_right_point.emplace_back(polygon.size(),false);
        for (std::size_t j=0;j<polygon.size();++j) {
            const auto vector=polygon[(j+1)%polygon.size()]-polygon[j];
            pieces.gra_p_vector.back().push_back(polygon[j]-center);
            line_data line{};
            line.length=cv::norm(vector);
            line.line_vector=vector;
            line.start_point=j;
            line.end_point=(j+1)%polygon.size();
            line.near_line={-1,-1};
            pieces.pieces_lines.back().push_back(line);
        }
        CardFragment fragment;
        fragment.source_index=id;
        fragment.bounds=cv::boundingRect(contour);
        cv::Mat full_mask=cv::Mat::zeros(warped.size(),CV_8UC1);
        // 只填外轮廓：内部的黑字和花纹仍属于同一块实体纸片。
        cv::drawContours(full_mask,std::vector<std::vector<cv::Point>>{contour},0,cv::Scalar(255),cv::FILLED);
        cv::bitwise_or(recognition.foreground_mask,full_mask,recognition.foreground_mask);
        fragment.mask=full_mask(fragment.bounds).clone();
        fragment.image=warped(fragment.bounds).clone();
        fragment.texture_mask=fragment.mask.clone();
        std::vector<cv::Point> local_model;
        for (const auto point : polygon) local_model.emplace_back(cvRound(point.x)-fragment.bounds.x,
                                                                  cvRound(point.y)-fragment.bounds.y);
        cv::Mat model_mask=cv::Mat::zeros(fragment.mask.size(),CV_8UC1);
        cv::fillPoly(model_mask,std::vector<std::vector<cv::Point>>{local_model},cv::Scalar(255));
        // Extend only texture to the fitted cut line. Leave every vertex region
        // (including original rounded corners), physical mask and pickup center intact.
        for (const auto point : local_model) cv::circle(model_mask,point,4*scale,cv::Scalar(0),cv::FILLED);
        cv::bitwise_or(fragment.texture_mask,model_mask,fragment.texture_mask);
        std::vector<int> whites;
        for (int y=0;y<fragment.image.rows;y+=3)
            for (int x=0;x<fragment.image.cols;x+=3)
                if (fragment.mask.at<uchar>(y,x)) {
                    const auto color=fragment.image.at<cv::Vec3b>(y,x);
                    whites.push_back(std::min({color[0],color[1],color[2]}));
                }
        std::sort(whites.begin(),whites.end());
        const double white=whites.empty() ? 255 : std::max(80,whites[whites.size()*9/10]);
        fragment.normalized_ink=cv::Mat(fragment.image.size(),CV_32FC1);
        for (int y=0;y<fragment.image.rows;++y)
            for (int x=0;x<fragment.image.cols;++x) {
                const auto color=fragment.image.at<cv::Vec3b>(y,x);
                fragment.normalized_ink.at<float>(y,x)=std::clamp(1-std::min({color[0],color[1],color[2]})/white,0.0,1.0);
            }
        recognition.fragments.push_back(std::move(fragment));
        cv::polylines(recognition.debug_image,std::vector<std::vector<cv::Point>>{contour},true,{0,255,0},1,cv::LINE_AA);
        for (std::size_t j=0;j<polygon.size();++j)
            cv::line(recognition.debug_image,polygon[j],polygon[(j+1)%polygon.size()],{255,200,0},1,cv::LINE_AA);
        cv::circle(recognition.debug_image,center,4,{0,0,255},cv::FILLED);
        cv::putText(recognition.debug_image,std::to_string(id),center+cv::Point2f(6,-6),
                    cv::FONT_HERSHEY_SIMPLEX,0.7,{0,0,255},2);
    }
    recognition.ready=!pieces.pieces_point.empty();
    if (!recognition.ready) recognition.error="未识别到分离的扑克牌碎片，请查看 task2_mask.png";
    return recognition;
}

double card_seam_score(const CardRecognition& recognition,const PuzzlePlacement& a,
                       const PuzzlePlacement& b,double tolerance) {
    if (a.points.size()<3 || b.points.size()<3 || a.source_index==b.source_index) return 0;
    const double sign_a=cv::contourArea(a.points,true)>0 ? 1 : -1;
    const double sign_b=cv::contourArea(b.points,true)>0 ? 1 : -1;
    double score=0;
    for (std::size_t i=0;i<a.points.size();++i) {
        const auto p=a.points[i],v=a.points[(i+1)%a.points.size()]-p;
        const double length=cv::norm(v);
        if (length<8) continue;
        const auto direction=v/static_cast<float>(length);
        const cv::Point2f inward_a(-direction.y*sign_a,direction.x*sign_a);
        for (std::size_t j=0;j<b.points.size();++j) {
            const auto q=b.points[j],w=b.points[(j+1)%b.points.size()]-q;
            const double other_length=cv::norm(w);
            if (other_length<8) continue;
            const auto other=w/static_cast<float>(other_length);
            const cv::Point2f inward_b(-other.y*sign_b,other.x*sign_b);
            if (std::abs(direction.x*other.y-direction.y*other.x)>0.10 || inward_a.dot(inward_b)>-0.9) continue;
            const double start=direction.dot(q-p),end=direction.dot(q+w-p);
            const double lo=std::max(0.0,std::min(start,end)),hi=std::min(length,std::max(start,end));
            if (hi-lo<8) continue;
            const auto midpoint=p+direction*static_cast<float>((lo+hi)*0.5);
            const double distance=std::abs((midpoint-q).x*other.y-(midpoint-q).y*other.x);
            if (distance>tolerance*2+1) continue;
            const int count=std::clamp(static_cast<int>((hi-lo)/4),8,64);
            int valid=0;
            double accumulated=0;
            for (int k=0;k<count;++k) {
                const auto on_a=p+direction*static_cast<float>(lo+(hi-lo)*(k+0.5)/count);
                const auto on_b=q+other*other.dot(on_a-q);
                cv::Vec3d ca,cb;
                if (!sample(recognition,a,on_a+inward_a*2.0f,ca) ||
                    !sample(recognition,b,on_b+inward_b*2.0f,cb)) continue;
                const double mismatch=std::min(1.0,cv::norm(ca-cb)/(255*std::sqrt(3.0)));
                const double ink_a=1-(ca[0]+ca[1]+ca[2])/(3*255);
                const double ink_b=1-(cb[0]+cb[1]+cb[2])/(3*255);
                accumulated+=0.1*(1-mismatch)+2*std::min(ink_a,ink_b)*(1-mismatch)-2*std::abs(ink_a-ink_b);
                ++valid;
            }
            if (valid>=count/2) score+=(hi-lo)*accumulated/valid;
        }
    }
    return score;
}

CardPuzzleResult solve_card_puzzle(const cv::Mat& warped,const CardPuzzleOptions& options) {
    CardPuzzleResult card;
    card.recognition=recognize_card_pieces(warped,options);
    if (!card.recognition.ready) { card.error=card.recognition.error; return card; }
    if (!std::isfinite(options.min_aspect) || !std::isfinite(options.max_aspect) ||
        options.min_aspect<=0 || options.max_aspect<options.min_aspect || options.max_aspect>1 ||
        !std::isfinite(options.gap_mm) || options.gap_mm<=0 ||
        !std::isfinite(options.target_size_mm.width) || !std::isfinite(options.target_size_mm.height) ||
        options.target_size_mm.width<0 || options.target_size_mm.height<0 ||
        ((options.target_size_mm.width==0)!=(options.target_size_mm.height==0))) {
        card.error="扑克牌目标尺寸、长宽比或机械间距无效";
        return card;
    }
    auto solver=options.solver;
    // 缓存相对姿态：整体平移/旋转不会改变两块的接缝图案匹配结果。
    using Key=std::tuple<std::size_t,std::size_t,long long,long long,long long>;
    std::map<Key,double> scores;
    solver.seam_score=[&](const PuzzlePlacement& a,const PuzzlePlacement& b) {
        const auto relative=rotate(b.translation-a.translation,-a.rotation_radians);
        const Key key{a.source_index,b.source_index,std::llround(std::remainder(b.rotation_radians-a.rotation_radians,2*CV_PI)*10000),
                      std::llround(relative.x*4),std::llround(relative.y*4)};
        const auto old=scores.find(key);
        if (old!=scores.end()) return old->second;
        const double score=card_seam_score(card.recognition,a,b,solver.position_tolerance);
        scores.emplace(key,score);
        return score;
    };
    if (options.use_white_border || options.use_rounded_corners) {
        const auto evidence=border_evidence(card.recognition);
        const auto corners=corner_evidence(card.recognition);
        solver.boundary_score=[&, evidence, corners](const PuzzlePlacement& placement,const cv::Rect2f& target) {
            double score=0;
            const double tolerance=solver.position_tolerance+1;
            for (std::size_t i=0;i<placement.points.size();++i) {
                const auto p=placement.points[i],q=placement.points[(i+1)%placement.points.size()];
                const bool on_border=(std::abs(p.x-target.x)<=tolerance && std::abs(q.x-target.x)<=tolerance) ||
                    (std::abs(p.x-target.br().x)<=tolerance && std::abs(q.x-target.br().x)<=tolerance) ||
                    (std::abs(p.y-target.y)<=tolerance && std::abs(q.y-target.y)<=tolerance) ||
                    (std::abs(p.y-target.br().y)<=tolerance && std::abs(q.y-target.br().y)<=tolerance);
                if (options.use_white_border && on_border)
                    score+=cv::norm(q-p)*evidence[placement.source_index][i];
                const double distance=std::min({cv::norm(p-target.tl()),cv::norm(p-target.br()),
                    cv::norm(p-cv::Point2f(target.x+target.width,target.y)),
                    cv::norm(p-cv::Point2f(target.x,target.y+target.height))});
                if (options.use_rounded_corners)
                    score+=300*corners[placement.source_index][i]*(1-std::min(2.0,distance/(2*tolerance)));
            }
            return score;
        };
    }
    if (options.use_layout_pattern || options.require_corner_indices)
        solver.layout_score=[&](const std::vector<PuzzlePlacement>& placements,const cv::Rect2f& target) {
            const auto layout=evaluate_card_layout(card.recognition,placements,target);
            return PuzzleLayoutScore{options.use_layout_pattern ? layout.score : 0,
                                     !options.require_corner_indices || layout.indices_verified};
        };
    const cv::Point2f origin(240,240);
    if (options.target_size_mm.area()>0) {
        card.result=solve_puzzle(card.recognition.pieces.pieces_point,
            cv::Rect2f(origin,options.target_size_mm*static_cast<float>(scale)),solver);
    } else {
        double area=0;
        for (const auto& p : card.recognition.pieces.pieces_point) area+=std::abs(cv::contourArea(p));
        const float min_width=std::sqrt(area*options.min_aspect)*0.985;
        const float max_width=std::sqrt(area*options.max_aspect)*1.015;
        const float min_height=std::sqrt(area/options.max_aspect)*0.985;
        const float max_height=std::sqrt(area/options.min_aspect)*1.015;
        card.result=solve_puzzle_in_range(card.recognition.pieces.pieces_point,origin,
            {min_width,min_height},{max_width,max_height},solver);
    }
    if (!card.result.solved) {
        card.error=card.result.search_limit_reached ? "扑克牌复原搜索达到上限" : "未找到满足几何约束的完整牌面";
        return card;
    }
    card.layout=evaluate_card_layout(card.recognition,card.result.placements,card.result.target);
    if (options.require_corner_indices && !card.layout.indices_verified) {
        card.assembled_image=render(card.recognition,card.result.placements);
        if (!card.layout.first_index || !card.layout.opposite_index) {
            const std::string location=!card.layout.first_index && !card.layout.opposite_index ? "左上和右下" :
                                       !card.layout.first_index ? "左上" : "右下";
            card.error=location+"角标未提取到独立的数字/字母与小花色，请检查清晰度后重新拍照";
        } else if (card.layout.rank_match<0.65) {
            card.error="对角角标的数字/字母形状匹配不足，请检查拼接预览后重新拍照";
        } else if (card.layout.suit_match<0.65) {
            card.error="对角角标的小花色形状匹配不足，请检查拼接预览后重新拍照";
        } else {
            card.error="对角角标的数字/字母宽高比例不一致，请检查拼接预览后重新拍照";
        }
        card.result.solved=false;
        return card;
    }
    // 轮廓模型为圆角使用了直边交点；机械搬运仍跟随实体纸片的面积重心。
    for (auto& placement : card.result.placements)
        placement.center_gravity=rotate(card.recognition.pieces.center_gravity[placement.source_index],
                                        placement.rotation_radians)+placement.translation;
    // Keep every expanded card piece above the actual warped A4 midpoint.
    // Keep 12 mm above the centerline: 10 mm higher than the previous layout,
    // plus the existing 2 mm allowance for rasterization and contour-fit error.
    const cv::Rect2f machine_workspace(0,0,static_cast<float>(warped.cols),
                                      warped.rows*0.5f-12*scale);
    card.result_machine=make_machine_result(card.result,options.gap_mm*scale,machine_workspace);
    card.assembled_image=render(card.recognition,card.result.placements);
    if (card.result_machine.ready) card.machine_image=render(card.recognition,card.result_machine.placements);
    else card.error=card.result_machine.error;
    return card;
}

namespace {
struct IndexSignature {
    bool found=false;
    cv::Mat rank,suit;
    double aspect=0;
};

cv::Mat normalized_symbol(const cv::Mat& symbol,cv::Size size) {
    cv::Mat result=cv::Mat::zeros(size,CV_32FC1),resized;
    const double factor=std::min((size.width-4.0)/symbol.cols,(size.height-4.0)/symbol.rows);
    cv::resize(symbol,resized,{std::max(1,cvRound(symbol.cols*factor)),std::max(1,cvRound(symbol.rows*factor))});
    resized.convertTo(resized,CV_32F,symbol.depth()==CV_32F ? 1 : 1.0/255);
    resized.copyTo(result(cv::Rect((size.width-resized.cols)/2,(size.height-resized.rows)/2,resized.cols,resized.rows)));
    cv::GaussianBlur(result,result,{0,0},0.7);
    // Compare glyph shape independently of local printed-ink contrast.
    cv::normalize(result,result,1.0,0,cv::NORM_INF);
    return result;
}

IndexSignature find_index_stack(const cv::Mat& contrast,const cv::Mat& source,bool separate_thin_lines) {
    cv::Mat binary=source.clone(),labels,stats,centers;
    const double sampling=binary.cols/24.0;
    // Thin fitted-cut gaps and court-frame lines must not merge into the small suit.
    // If no stack can be found, also separate thin bridges from the rank: a
    // blurred Q tail can touch the neighboring court frame. Keep the original
    // contrast image for comparing glyphs, including their thin strokes.
    cv::Mat separation=separate_thin_lines ? binary : binary.rowRange(cvRound(30*sampling),binary.rows);
    cv::morphologyEx(separation,separation,cv::MORPH_OPEN,
                    cv::getStructuringElement(cv::MORPH_ELLIPSE,{cvRound(2*sampling)|1,cvRound(2*sampling)|1}));
    const int count=cv::connectedComponentsWithStats(binary,labels,stats,centers);
    IndexSignature best;
    int best_area=0;
    for (int suit=1;suit<count;++suit) {
        const cv::Rect sb(stats.at<int>(suit,cv::CC_STAT_LEFT),stats.at<int>(suit,cv::CC_STAT_TOP),
                          stats.at<int>(suit,cv::CC_STAT_WIDTH),stats.at<int>(suit,cv::CC_STAT_HEIGHT));
        const int suit_area=stats.at<int>(suit,cv::CC_STAT_AREA);
        if (suit_area<7*sampling*sampling || sb.width<4*sampling || sb.width>15*sampling ||
            sb.height<5*sampling || sb.height>15*sampling || sb.x<sampling ||
            sb.br().x>=binary.cols-sampling || sb.y<22*sampling || sb.y>43*sampling) continue;
        cv::Rect rb;
        int rank_area=0;
        for (int rank=1;rank<count;++rank) {
            const cv::Rect box(stats.at<int>(rank,cv::CC_STAT_LEFT),stats.at<int>(rank,cv::CC_STAT_TOP),
                               stats.at<int>(rank,cv::CC_STAT_WIDTH),stats.at<int>(rank,cv::CC_STAT_HEIGHT));
            const int area=stats.at<int>(rank,cv::CC_STAT_AREA);
            if (area<2*sampling*sampling || box.y<sampling || box.br().y>=sb.y || box.x<sampling ||
                box.br().x>=binary.cols-sampling ||
                std::abs(centers.at<double>(rank,0)-centers.at<double>(suit,0))>7*sampling) continue;
            rb=rank_area ? rb|box : box;
            rank_area+=area;
        }
        if (rank_area<10*sampling*sampling || rb.y>15*sampling || rb.height<10*sampling ||
            rb.height>29*sampling || rb.width<4*sampling || rb.width>21*sampling ||
            sb.y-rb.br().y>14*sampling || rb.width>rb.height*2.2 || rank_area+suit_area<=best_area) continue;
        best.found=true;
        best.aspect=static_cast<double>(rb.width)/rb.height;
        best.rank=normalized_symbol(contrast(rb),{24,32});
        best.suit=normalized_symbol(contrast(sb),{24,24});
        best_area=rank_area+suit_area;
    }
    return best;
}

// Locate the rank-over-small-suit stack, without a font model or a hard-coded
// card value. Two-digit ranks may contain several connected components.
IndexSignature index_signature(const cv::Mat& patch) {
    const double sampling=patch.cols/24.0;
    std::vector<float> levels;
    for (int y=0;y<cvRound(30*sampling);++y)
        for (int x=0;x<cvRound(patch.cols*0.8);++x) levels.push_back(patch.at<float>(y,x));
    std::sort(levels.begin(),levels.end());
    const double paper=levels[levels.size()/5],peak=levels[levels.size()*98/100];
    if (peak-paper<0.10) return {};
    cv::Mat contrast,binary;
    cv::subtract(patch,paper,contrast);
    cv::max(contrast,0,contrast);
    contrast/=peak-paper;
    cv::min(contrast,1,contrast);
    cv::compare(contrast,0.25,binary,cv::CMP_GT);
    for (bool separate : {false,true}) {
        const auto signature=find_index_stack(contrast,binary,separate);
        if (signature.found) return signature;
    }
    // A blurred court-frame rule can join BOTH glyphs with bridges wider than
    // the opening kernel. Only as a fallback, locate a long vertical ink rule
    // on the inner side of the index. Require support down to the patch bottom
    // (below the small suit), so a wide rank such as "10" is not itself a rule.
    // Remove the rule and its inner side from the localization mask only.
    // Glyph comparison still samples the original continuous contrast image.
    const int top=cvRound(patch.rows*0.2),bottom=cvRound(patch.rows*0.9);
    for (int x=cvRound(patch.cols*0.7);x<patch.cols-sampling;++x) {
        if (cv::countNonZero(binary.col(x).rowRange(top,patch.rows))<0.7*(patch.rows-top) ||
            cv::countNonZero(binary.col(x).rowRange(bottom,patch.rows))<0.7*(patch.rows-bottom)) continue;
        cv::Mat without_frame=binary.clone();
        without_frame.colRange(std::max(0,x-cvRound(sampling/2)),patch.cols).setTo(0);
        for (bool separate : {false,true}) {
            const auto signature=find_index_stack(contrast,without_frame,separate);
            if (signature.found) return signature;
        }
        break;
    }
    return {};
}

double ink_agreement(const cv::Mat& a,const cv::Mat& b) {
    const double sum=cv::sum(a)[0]+cv::sum(b)[0];
    if (sum<1) return 0;
    cv::Mat common;
    cv::min(a,b,common);
    return 2*cv::sum(common)[0]/sum;
}
}

CardLayoutEvidence evaluate_card_layout(const CardRecognition& recognition,
    const std::vector<PuzzlePlacement>& placements,const cv::Rect2f& target) {
    CardLayoutEvidence result;
    if (target.width<=0 || target.height<=0) return result;
    // A small face canvas makes complete-layout comparisons cheap. Whiteness is
    // normalized per physical fragment, so illumination does not dominate ink.
    const cv::Size size(240,360);
    cv::Mat face=cv::Mat::zeros(size,CV_32FC1);
    const double sx=size.width/target.width,sy=size.height/target.height;
    for (const auto& placement : placements) {
        const auto& fragment=recognition.fragments.at(placement.source_index);
        const double c=std::cos(placement.rotation_radians),s=std::sin(placement.rotation_radians);
        const cv::Matx23d transform(sx*c,-sx*s,sx*(c*fragment.bounds.x-s*fragment.bounds.y+placement.translation.x-target.x),
                                   sy*s,sy*c,sy*(s*fragment.bounds.x+c*fragment.bounds.y+placement.translation.y-target.y));
        cv::Mat ink,mask;
        cv::warpAffine(fragment.normalized_ink,ink,transform,size,cv::INTER_LINEAR);
        cv::warpAffine(fragment.texture_mask.empty() ? fragment.mask : fragment.texture_mask,
                       mask,transform,size,cv::INTER_NEAREST);
        ink.copyTo(face,mask);
    }
    const cv::Rect index_zone(4,6,48,106);
    cv::Mat opposite;
    cv::flip(face(cv::Rect(size.width-index_zone.br().x,size.height-index_zone.br().y,
                           index_zone.width,index_zone.height)),opposite,-1);
    const auto first=index_signature(face(index_zone)),second=index_signature(opposite);
    result.first_index=first.found;
    result.opposite_index=second.found;
    if (first.found && second.found) {
        const double rank=ink_agreement(first.rank,second.rank),suit=ink_agreement(first.suit,second.suit);
        result.rank_match=rank;result.suit_match=suit;
        result.index_match=0.65*rank+0.35*suit;
        const double aspect=first.aspect/second.aspect;
        result.index_aspect_ratio=aspect;
        result.indices_verified=rank>=0.65 && suit>=0.65 && aspect>=0.6 && aspect<=1.65;
    }
    // Number-card pips and the ace motif often have bilateral structure; court
    // figures commonly repeat after a half-turn. Compare both, excluding indices.
    cv::Mat body,flipped;
    cv::resize(face(cv::Rect(46,10,148,340)),body,{74,170},0,0,cv::INTER_AREA);
    cv::subtract(body,0.2,body);
    cv::max(body,0,body);body/=0.6;cv::min(body,1,body);
    cv::GaussianBlur(body,body,{0,0},0.8);
    for (const int axis : {-1,1}) {
        cv::flip(body,flipped,axis);
        result.pattern_match=std::max(result.pattern_match,ink_agreement(body,flipped));
    }
    result.score=700*result.index_match+(result.indices_verified ? 250 : 0)+200*result.pattern_match;
    if (first.found!=second.found) result.score-=150;
    return result;
}
