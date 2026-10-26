#include "card_puzzle.hpp"

#include <iostream>
#include <stdexcept>

namespace {
void require(bool value,const char* message) {
    if (!value) throw std::runtime_error(message);
}

CardLayoutEvidence index_evidence(const cv::Mat& ink) {
    CardRecognition recognition;
    CardFragment fragment;
    fragment.bounds={0,0,240,360};
    fragment.mask=cv::Mat(ink.size(),CV_8UC1,cv::Scalar(255));
    fragment.texture_mask=fragment.mask;
    fragment.normalized_ink=ink;
    recognition.fragments.push_back(fragment);
    return evaluate_card_layout(recognition,std::vector<PuzzlePlacement>{PuzzlePlacement{}},{0,0,240,360});
}

void verify_recorded_index_stacks() {
#ifdef CARD_INDEX_STACK_FIXTURE
    // Actual normalized ink from the nine saved layouts, with their original
    // rotations and subpixel interpolation. Seven previously lost an index to
    // the adjacent court frame; JPEG re-recognition can hide this exact defect.
    cv::FileStorage fixtures(CARD_INDEX_STACK_FIXTURE,cv::FileStorage::READ);
    require(fixtures.isOpened(),"missing recorded index stacks");
    require(fixtures["captures"].size()==9,"lost an index capture regression");
    for (const auto& node : fixtures["captures"]) {
        cv::Mat ink=cv::Mat::zeros(360,240,CV_32FC1),first,opposite;
        node["first"]>>first;node["opposite"]>>opposite;
        first.copyTo(ink(cv::Rect(4,6,48,106)));
        opposite.copyTo(ink(cv::Rect(188,248,48,106)));
        const auto evidence=index_evidence(ink);
        if (!evidence.indices_verified) std::cerr << "recorded index " << std::string(node["capture"]) << '\n';
        require(evidence.indices_verified,"court-frame bridges hid matching recorded Q indices");
        require(evidence.rank_match>=0.65 && evidence.suit_match>=0.65,
                "recorded index verification thresholds weakened");
        ink(cv::Rect(188,248,48,106)).setTo(0);
        require(!index_evidence(ink).indices_verified,"frame fallback accepted a missing opposite index");
        first.copyTo(ink(cv::Rect(188,248,48,106)));
        require(!index_evidence(ink).indices_verified,"frame fallback accepted indices in the same direction");
        opposite.copyTo(ink(cv::Rect(188,248,48,106)));
        // Keep the real court frame and small club, but replace the Q with A.
        // Recovering the frame must not bypass comparison of the two ranks.
        ink(cv::Rect(12,15,31,51)).setTo(0);
        cv::putText(ink,"A",{14,61},cv::FONT_HERSHEY_SIMPLEX,1.8,cv::Scalar(1),3);
        require(!index_evidence(ink).indices_verified,"frame fallback accepted different corner ranks");
    }
#endif
}

void verify_all_generated_indices() {
    const std::vector<std::string> ranks={"A","2","3","4","5","6","7","8","9","10","J","Q","K"};
    for (const auto& rank : ranks) for (int suit=0;suit<4;++suit) {
        cv::Mat face(360,240,CV_8UC3,cv::Scalar(255,255,255));
        const cv::Scalar color=suit%2 ? cv::Scalar(0,0,210) : cv::Scalar(0,0,0);
        cv::Mat glyph(64,90,CV_8UC3,cv::Scalar(255,255,255));
        int baseline=0;
        const auto text_size=cv::getTextSize(rank,cv::FONT_HERSHEY_SIMPLEX,1.6,2,&baseline);
        cv::putText(glyph,rank,{2,44},cv::FONT_HERSHEY_SIMPLEX,1.6,color,2,cv::LINE_AA);
        cv::Mat cropped=glyph(cv::Rect(1,44-text_size.height-1,text_size.width+2,text_size.height+3)),resized;
        cv::resize(cropped,resized,{rank=="10" ? 34 : 25,42});
        resized.copyTo(face(cv::Rect(14,17,resized.cols,resized.rows)));
        if (suit==0) {
            for (const auto p : std::vector<cv::Point>{{26,81},{20,88},{32,88}}) cv::circle(face,p,6,color,cv::FILLED);
            cv::fillConvexPoly(face,std::vector<cv::Point>{{26,86},{22,98},{30,98}},color);
        } else if (suit==1) {
            cv::circle(face,{21,82},6,color,cv::FILLED);cv::circle(face,{31,82},6,color,cv::FILLED);
            cv::fillConvexPoly(face,std::vector<cv::Point>{{15,82},{37,82},{26,98}},color);
        } else if (suit==2) {
            cv::fillConvexPoly(face,std::vector<cv::Point>{{26,76},{15,89},{37,89}},color);
            cv::circle(face,{21,88},6,color,cv::FILLED);cv::circle(face,{31,88},6,color,cv::FILLED);
            cv::fillConvexPoly(face,std::vector<cv::Point>{{26,88},{22,99},{30,99}},color);
        } else cv::fillConvexPoly(face,std::vector<cv::Point>{{26,75},{16,87},{26,99},{36,87}},color);
        cv::Mat opposite;
        cv::flip(face(cv::Rect(0,0,60,110)),opposite,-1);
        opposite.copyTo(face(cv::Rect(180,250,60,110)));
        auto evidence=[&](const cv::Mat& image) {
            CardRecognition recognized;
            CardFragment fragment;
            fragment.bounds={0,0,240,360};fragment.image=image;
            fragment.mask=cv::Mat(image.size(),CV_8UC1,cv::Scalar(255));fragment.texture_mask=fragment.mask;
            fragment.normalized_ink=cv::Mat(image.size(),CV_32FC1);
            for (int y=0;y<image.rows;++y) for (int x=0;x<image.cols;++x) {
                const auto pixel=image.at<cv::Vec3b>(y,x);
                fragment.normalized_ink.at<float>(y,x)=1-std::min({pixel[0],pixel[1],pixel[2]})/255.f;
            }
            recognized.fragments.push_back(fragment);
            return evaluate_card_layout(recognized,std::vector<PuzzlePlacement>{PuzzlePlacement{}},{0,0,240,360});
        };
        const auto good=evidence(face);
        if (!good.indices_verified) std::cerr << "generated " << rank << " suit " << suit << " missing index\n";
        require(good.indices_verified,"a generated rank/suit pair failed diagonal verification");
        cv::Mat faded=face.clone();
        cv::Mat faded_index=faded(cv::Rect(0,0,60,110));
        faded_index.convertTo(faded_index,-1,0.35,255*0.65);
        require(evidence(faded).indices_verified,"faint but matching corner strokes were rejected");
        cv::Mat same_direction=face.clone();
        face(cv::Rect(0,0,60,110)).copyTo(same_direction(cv::Rect(180,250,60,110)));
        require(!evidence(same_direction).indices_verified,"upright indices on both corners were accepted");
        face(cv::Rect(180,250,60,110)).setTo(cv::Scalar(255,255,255));
        require(!evidence(face).indices_verified,"a missing opposite index was accepted");
    }
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
        verify_all_generated_indices();
        verify_recorded_index_stacks();
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
        require(std::abs(cv::contourArea(rounded_piece.pieces.pieces_point.front())-60000)<1200,
                "near-right-angle rounded seeds bypassed supported straight-edge fitting");
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
#ifdef CARD_ACE_PHOTO_FIXTURE
        // Sparse black printing used to put both A corners on one side and
        // scatter the central spade, despite passing rectangle coverage checks.
        const auto ace_photo=cv::imread(CARD_ACE_PHOTO_FIXTURE);
        require(!ace_photo.empty(),"missing spade-A fixture");
        const auto ace_result=solve_card_puzzle(ace_photo);
        verify(ace_result,"four-piece spade A");
        require(ace_result.recognition.fragments.size()==4,"spade-A piece count changed");
        const auto ace_target=ace_result.result.target;
        const auto ace_middle=ace_target.tl()+cv::Point2f(ace_target.width/2,ace_target.height/2);
        std::vector<cv::Point2f> ace_centers(4);
        const std::vector<std::size_t> physical_corners={3,3,1,3};
        for (const auto& placement : ace_result.result.placements) {
            ace_centers[placement.source_index]=placement.center_gravity;
            const auto vertex=placement.points[physical_corners[placement.source_index]];
            const double distance=std::min({cv::norm(vertex-ace_target.tl()),cv::norm(vertex-ace_target.br()),
                cv::norm(vertex-cv::Point2f(ace_target.x+ace_target.width,ace_target.y)),
                cv::norm(vertex-cv::Point2f(ace_target.x,ace_target.y+ace_target.height))});
            require(distance<8,"spade-A physical rounded corner placed on an internal seam");
        }
        require((ace_centers[1].x-ace_middle.x)*(ace_centers[3].x-ace_middle.x)<0 &&
                (ace_centers[1].y-ace_middle.y)*(ace_centers[3].y-ace_middle.y)<0,
                "spade-A printed corner indices not restored to opposite diagonals");
#endif
#ifdef CARD_SPLIT_CUT_PHOTO_FIXTURE
        // A printed club split one cut into two near-collinear fitted lines;
        // intersecting them used to fall back to a noisy nine-vertex contour.
        const auto split_cut_photo=cv::imread(CARD_SPLIT_CUT_PHOTO_FIXTURE);
        require(!split_cut_photo.empty(),"missing split-cut club-six fixture");
        const auto split_cut=recognize_card_pieces(split_cut_photo);
        require(split_cut.ready && split_cut.fragments.size()==4,"split-cut photo lost a physical card piece");
        for (const auto& polygon : split_cut.pieces.pieces_point)
            require(polygon.size()==4,"two supported parts of one straight cut were not merged");
#endif
#ifdef CARD_SHORT_CUT_PHOTO_FIXTURE
        // A real short straight cut has rounded/noisy endpoints; one seed chord
        // was discarded as a flower arc, producing a nine-edge model.
        const auto short_cut_photo=cv::imread(CARD_SHORT_CUT_PHOTO_FIXTURE);
        require(!short_cut_photo.empty(),"missing noisy short-cut fixture");
        const auto short_cut=recognize_card_pieces(short_cut_photo);
        require(short_cut.ready && short_cut.fragments.size()==4,"short-cut capture lost a card piece");
        for (const auto& polygon : short_cut.pieces.pieces_point)
            require(polygon.size()==4,"supported short straight cut became spurious polygon edges");
#endif
#ifdef CARD_SIX_PHOTO_FIXTURE
        const auto six_photo=cv::imread(CARD_SIX_PHOTO_FIXTURE);
        CardPuzzleOptions indexed_options;
        indexed_options.require_corner_indices=true;
        const auto six_result=solve_card_puzzle(six_photo,indexed_options);
        verify(six_result,"four-piece club six");
        require(six_result.layout.indices_verified,"club-six indices not on opposite corners and directions");
        const double center_line=six_photo.rows*0.5-10*scale;
        for (const auto& placement : six_result.result_machine.placements)
            for (const auto point : placement.points)
                require(point.y<center_line,"card machine polygon extends below the A4 center line");
        cv::Mat machine_gray;
        cv::cvtColor(six_result.machine_image,machine_gray,cv::COLOR_BGR2GRAY);
        std::vector<cv::Point> painted;
        cv::findNonZero(machine_gray,painted);
        require(!painted.empty() && cv::boundingRect(painted).br().y<=center_line,
                "physical card texture extends below the A4 center line");
        require(six_result.layout.pattern_match>0.7,"club-six inner pips are no longer aligned");
        for (const auto& fragment : six_result.recognition.fragments) {
            cv::Mat recovered;
            cv::subtract(fragment.texture_mask,fragment.mask,recovered);
            require(cv::countNonZero(recovered)>100,"cut-edge printed pips missing from texture mask");
        }
        auto wrong=six_result.result.placements;
        // Move a single indexed quarter through a half-turn in its own slot.
        for (auto& placement : wrong) if (placement.source_index==1) {
            placement.rotation_radians+=CV_PI;
            placement.translation=placement.center_gravity*2-placement.translation;
        }
        require(!evaluate_card_layout(six_result.recognition,wrong,six_result.result.target).indices_verified,
                "wrongly oriented club-six index passed the final send guard");
#endif
        if (argc>1) {
#ifdef CARD_SIX_PHOTO_FIXTURE
            cv::imwrite(std::string(argv[1])+"/task2_club_six_result.png",six_result.assembled_image);
#endif
#ifdef CARD_FOUR_PHOTO_FIXTURE
            cv::imwrite(std::string(argv[1])+"/task2_heart_j_result.png",four_photo_result.assembled_image);
            cv::imwrite(std::string(argv[1])+"/task2_heart_j_machine.png",four_photo_result.machine_image);
#endif
#ifdef CARD_ROTATED_PHOTO_FIXTURE
            cv::imwrite(std::string(argv[1])+"/task2_rotated_heart_j_result.png",rotated_result.assembled_image);
#endif
#ifdef CARD_ACE_PHOTO_FIXTURE
            cv::imwrite(std::string(argv[1])+"/task2_spade_ace_result.png",ace_result.assembled_image);
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
