#include "card_puzzle.hpp"

#include <iostream>
#include <stdexcept>

namespace {
void require(bool value,const char* message) {
    if (!value) throw std::runtime_error(message);
}

cv::Mat face() {
    cv::Mat card(336,240,CV_8UC3,cv::Scalar(240,240,240));
    cv::line(card,{0,62},{239,62},{30,30,210},12);
    cv::line(card,{0,210},{239,210},{40,40,170},20);
    cv::line(card,{75,0},{75,335},{25,25,200},10);
    cv::circle(card,{40,135},14,{3,3,3},cv::FILLED);
    cv::putText(card,"K",{160,290},cv::FONT_HERSHEY_SIMPLEX,1.3,{5,5,5},4);
    return card;
}

cv::Mat two_pieces() {
    const auto card=face();
    cv::Mat scene(1178,830,CV_8UC3,cv::Scalar(25,25,25));
    card(cv::Rect(0,0,120,336)).copyTo(scene(cv::Rect(100,100,120,336)));
    cv::Mat flipped;
    cv::rotate(card(cv::Rect(120,0,120,336)),flipped,cv::ROTATE_180);
    flipped.copyTo(scene(cv::Rect(450,150,120,336)));
    return scene;
}

void verify(const CardPuzzleResult& card, const char* stage) {
    if (!card.result.solved) std::cerr << stage << ": " << card.error << '\n';
    require(card.result.solved,"card puzzle not solved");
    require(card.result_machine.ready,"machine result not ready");
    require(card.result.placements.size()==card.recognition.fragments.size(),"lost card piece");
    require(card.result_machine.minimum_gap_pixels>=20,"machine gap less than 5 mm");
    require(!card.assembled_image.empty() && !card.machine_image.empty(),"texture previews missing");
    for (const auto& p : card.result.placements) {
        const auto& source=card.recognition.pieces.pieces_point[p.source_index];
        const auto source_center=card.recognition.pieces.center_gravity[p.source_index];
        const double c=std::cos(p.rotation_radians),s=std::sin(p.rotation_radians);
        const cv::Point2f expected(c*source_center.x-s*source_center.y+p.translation.x,
                                    s*source_center.x+c*source_center.y+p.translation.y);
        require(cv::norm(expected-p.center_gravity)<0.002,"physical centroid not carried through rigid transform");
        for (std::size_t i=0;i<source.size();++i)
            for (std::size_t j=0;j<source.size();++j)
                require(std::abs(cv::norm(source[i]-source[j])-cv::norm(p.points[i]-p.points[j]))<0.005,
                        "piece geometry scaled or deformed");
    }
}
}

int main(int argc,char** argv) {
    try {
        const auto scene=two_pieces();
        const auto original=scene.clone();
        const auto recognized=recognize_card_pieces(scene);
        require(recognized.ready && recognized.fragments.size()==2,"printing split a physical card piece");
        require(recognized.foreground_mask.at<uchar>(235,140)==255,"black printing treated as a hole");
        require(recognized.fragments[0].image.at<cv::Vec3b>(135,40)[0]<10,"black printing erased from texture");
        require(cv::norm(scene,original,cv::NORM_INF)==0,"source image changed");
        for (std::size_t i=0;i<recognized.fragments.size();++i) {
            const auto& fragment=recognized.fragments[i];
            const auto m=cv::moments(fragment.mask,true);
            const cv::Point2f expected(m.m10/m.m00+fragment.bounds.x,m.m01/m.m00+fragment.bounds.y);
            require(cv::norm(recognized.pieces.center_gravity[i]-expected)<0.15,
                    "source centroid missing or incorrect");
        }
        cv::Mat dim;
        scene.convertTo(dim,-1,0.45,5);
        require(recognize_card_pieces(dim).fragments.size()==2,"weak-light colored card recognition failed");
        cv::Mat concave(600,600,CV_8UC3,cv::Scalar(25,25,25));
        cv::fillPoly(concave,std::vector<std::vector<cv::Point>>{{{100,100},{300,100},{300,180},{180,180},{180,300},{100,300}}},
                     cv::Scalar(235,235,235));
        const auto l_piece=recognize_card_pieces(concave);
        require(l_piece.ready && !cv::isContourConvex(l_piece.pieces.pieces_point.front()),"real concave cut replaced by convex hull");
        require(!recognize_card_pieces(cv::Mat{}).ready,"empty image accepted");
        require(!recognize_card_pieces(cv::Mat(600,600,CV_8UC3,cv::Scalar(25,25,25))).ready,"empty black paper accepted");
        cv::Mat cut_ink(600,600,CV_8UC3,cv::Scalar(30,30,30));
        cv::rectangle(cut_ink,{100,100},{300,450},{235,235,235},cv::FILLED);
        cv::Mat printed_region=cut_ink(cv::Rect(100,100,201,351));
        cv::circle(printed_region,{200,150},14,{2,2,2},cv::FILLED);
        const auto ink_repaired=recognize_card_pieces(cut_ink);
        require(ink_repaired.ready && ink_repaired.foreground_mask.at<uchar>(250,295)==255,
                "black ink at cut edge mistaken for missing card material");
        cv::rectangle(cut_ink,{280,320},{310,350},{30,30,30},cv::FILLED);
        const auto actual_cut=recognize_card_pieces(cut_ink);
        require(actual_cut.ready && actual_cut.foreground_mask.at<uchar>(335,290)==0,
                "real small concave cut filled as ink");
        cv::Mat rounded(600,600,CV_8UC3,cv::Scalar(25,25,25));
        cv::rectangle(rounded,{112,100},{288,400},{235,235,235},cv::FILLED);
        cv::rectangle(rounded,{100,112},{300,388},{235,235,235},cv::FILLED);
        for (const auto& center : std::vector<cv::Point>{{112,112},{288,112},{112,388},{288,388}})
            cv::circle(rounded,center,12,{235,235,235},cv::FILLED);
        const auto rounded_piece=recognize_card_pieces(rounded);
        require(rounded_piece.ready && rounded_piece.pieces.pieces_point.front().size()==4,
                "rounded card corners treated as extra cut edges");
        require(rounded_piece.foreground_mask.at<uchar>(100,100)==0,"rounded physical corner was filled");
        cv::Mat beveled(600,600,CV_8UC3,cv::Scalar(25,25,25));
        cv::fillPoly(beveled,std::vector<std::vector<cv::Point>>{{{100,100},{280,100},{300,120},{300,400},{100,400}}},
                     cv::Scalar(235,235,235));
        require(recognize_card_pieces(beveled).pieces.pieces_point.front().size()==5,
                "a real short straight bevel was removed as a rounded corner");

        CardPuzzleOptions options;
        options.target_size_mm={59.5f,83.75f};
        options.solver.time_limit_seconds=6;
        const auto fixed=solve_card_puzzle(scene,options);
        verify(fixed,"fixed-size synthetic");
        require(fixed.result.evaluated_solutions>1,"appearance search stopped at first geometric solution");
        const auto& p=fixed.result.placements;
        const double relative=std::abs(std::remainder(p[0].rotation_radians-p[1].rotation_radians,2*CV_PI));
        require(std::abs(relative-CV_PI)<0.04,"seam printing did not distinguish reversed half-card");
        options.target_size_mm={};
        const auto estimated=solve_card_puzzle(scene,options);
        verify(estimated,"automatic-size synthetic");
        require(std::abs(estimated.result.target.width-238)<6 && std::abs(estimated.result.target.height-335)<6,
                "automatic card dimensions incorrect");
        const auto full=face();
        cv::Mat four(1178,830,CV_8UC3,cv::Scalar(25,25,25));
        const std::vector<cv::Point> places={{90,100},{420,120},{100,600},{450,620}};
        for (int i=0;i<4;++i) {
            cv::Mat fragment=full(cv::Rect((i%2)*120,(i/2)*168,120,168)).clone();
            if (i%2) cv::rotate(fragment,fragment,cv::ROTATE_180);
            fragment.copyTo(four(cv::Rect(places[i],fragment.size())));
        }
        options.target_size_mm={59.5f,83.5f};
        options.solver.time_limit_seconds=8;
        const auto four_result=solve_card_puzzle(four,options);
        verify(four_result,"four-piece synthetic");
        require(four_result.result.placements.size()==4,"four-piece card recovery failed");
#ifdef CARD_PHOTO_FIXTURE
        const auto photo=cv::imread(CARD_PHOTO_FIXTURE);
        require(!photo.empty(),"missing uneven-light photo fixture");
        const auto photo_result=solve_card_puzzle(photo);
        verify(photo_result,"uneven-light photo");
        require(photo_result.recognition.fragments.size()==3,"photo piece count changed");
        const std::vector<std::size_t> expected_edges={5,3,4};
        for (std::size_t i=0;i<expected_edges.size();++i)
            require(photo_result.recognition.pieces.pieces_point[i].size()==expected_edges[i],"photo edge model incorrect");
        require(photo_result.recognition.foreground_mask.at<uchar>(1077,243)==0,
                "illuminated black background absorbed into card outline");
        require(photo_result.result.coverage>=0.97,"photo layout coverage below existing threshold");
#endif
#ifdef CARD_FOUR_PHOTO_FIXTURE
        // Four similar quadrilaterals used to swap the columns: the white frame
        // appeared down the middle while the printed figure was split outside.
        const auto four_photo=cv::imread(CARD_FOUR_PHOTO_FIXTURE);
        require(!four_photo.empty(),"missing four-piece heart-J fixture");
        const auto four_photo_result=solve_card_puzzle(four_photo);
        verify(four_photo_result,"four-piece heart-J photo");
        require(four_photo_result.recognition.fragments.size()==4,"four-piece photo count changed");
        const auto center=four_photo_result.result.target.tl()+cv::Point2f(
            four_photo_result.result.target.width/2,four_photo_result.result.target.height/2);
        for (const auto& placement : four_photo_result.result.placements) {
            const auto offset=placement.center_gravity-center;
            const double diagonal=offset.x*offset.y;
            // Invariant to a global 180-degree rotation of the whole card.
            const bool negative=placement.source_index==0 || placement.source_index==1;
            require(negative ? diagonal<0 : diagonal>0,
                    "heart-J columns swapped: white outside borders became an internal seam");
        }
        require(four_photo_result.result.boundary_score>0,"printed card edges selected as outside borders");
#endif
#ifdef CARD_ROTATED_PHOTO_FIXTURE
        // This arrangement timed out repeatedly because two rounded/straight
        // sides became extra seed edges. A solved rectangle must also restore
        // the two printed J corners to opposite diagonals.
        const auto rotated_photo=cv::imread(CARD_ROTATED_PHOTO_FIXTURE);
        require(!rotated_photo.empty(),"missing rotated heart-J fixture");
        const auto rotated_result=solve_card_puzzle(rotated_photo);
        verify(rotated_result,"rotated four-piece heart-J photo");
        require(rotated_result.recognition.fragments.size()==4,"rotated photo piece count changed");
        for (const auto& polygon : rotated_result.recognition.pieces.pieces_point)
            require(polygon.size()==4,"rounded/near-collinear edge introduced a spurious cut");
        std::vector<cv::Point2f> centers(4);
        for (const auto& placement : rotated_result.result.placements)
            centers[placement.source_index]=placement.center_gravity;
        const auto middle=rotated_result.result.target.tl()+cv::Point2f(
            rotated_result.result.target.width/2,rotated_result.result.target.height/2);
        require((centers[0].x-middle.x)*(centers[3].x-middle.x)<0 &&
                (centers[0].y-middle.y)*(centers[3].y-middle.y)<0,
                "rotated heart-J printing reversed inside a rectangular subassembly");
#endif
        if (argc>1) {
#ifdef CARD_FOUR_PHOTO_FIXTURE
            cv::imwrite(std::string(argv[1])+"/task2_heart_j_result.png",four_photo_result.assembled_image);
            cv::imwrite(std::string(argv[1])+"/task2_heart_j_machine.png",four_photo_result.machine_image);
#endif
#ifdef CARD_ROTATED_PHOTO_FIXTURE
            cv::imwrite(std::string(argv[1])+"/task2_rotated_heart_j_result.png",rotated_result.assembled_image);
#endif
            cv::imwrite(std::string(argv[1])+"/task2_synthetic_input.png",scene);
            cv::imwrite(std::string(argv[1])+"/task2_synthetic_result.png",fixed.assembled_image);
            cv::imwrite(std::string(argv[1])+"/task2_synthetic_machine.png",fixed.machine_image);
        }
        std::cout << "card puzzle tests passed; candidate score " << fixed.result.seam_score
                  << ", geometry-valid candidates " << fixed.result.evaluated_solutions << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
