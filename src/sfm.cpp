#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include "sfm.h"

namespace fs = std::filesystem;

SFM::SFM(std::string imgFolder, std::string maskFolder, std::string camerasPath) : 
    imgFolder(imgFolder), maskFolder(maskFolder), camerasPath(camerasPath){};

void SFM::computeSiftDescriptors(){
    std::cout << "ENTERED getSiftDescriptors\n";

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    fs::path imgFolderPath = imgFolder;
    fs::path maskFolderPath = maskFolder;

    std::cout << "imgFolder: " << imgFolderPath << '\n';
    std::cout << "maskFolder: " << maskFolderPath << '\n';
    std::cout << "current path: " << fs::current_path() << '\n';

    std::vector<std::string> imgPaths;
    std::vector<std::string> maskPaths; 
    // Iterate through image folder
    for (auto const& entry : fs::directory_iterator(imgFolderPath)){
        
        if (!entry.is_regular_file())
        continue;
        // Construct the corresponding mask file name
        fs::path maskPath = maskFolderPath / entry.path().filename().string().substr(3, 9);
        
        if (!fs::is_regular_file(maskPath))
            continue;

        imgPaths.push_back(entry.path().string());
        maskPaths.push_back(maskPath.string());
    }
    // Sort in alphabetical order
    std::sort(imgPaths.begin(), imgPaths.end());
    std::sort(maskPaths.begin(), maskPaths.end());
    imgFiles = imgPaths;

    // Knowing the number of files we can create the keypoints and descriptors vectors already with the right number of elements
    keypoints.assign(imgPaths.size(), {});
    descriptors.assign(imgPaths.size(), cv::Mat());

    // For each ffile in the folder collect keypoints and descriptors
    // Use sorted filenames
    for (int i = 0; i < imgPaths.size(); ++i) {
        cv::Mat img = cv::imread(imgPaths[i], cv::IMREAD_GRAYSCALE);
        cv::Mat mask = cv::imread(maskPaths[i], cv::IMREAD_GRAYSCALE);
    
        auto start = std::chrono::high_resolution_clock::now();

        sift -> detectAndCompute(img, mask, keypoints[i], descriptors[i]);

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed =
        std::chrono::duration<double, std::milli>(end - start);

        std::cout << "SIFT: " << elapsed.count()
          << " ms, keypoints: "
          << keypoints[i].size() << '\n';

        cv::Mat out;
        cv::drawKeypoints(img, keypoints[i], out, cv::Scalar::all(-1),
                        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        const fs::path outDir = fs::current_path() / "outputs" / "keypoints";
        fs::create_directories(outDir);

        fs::path outputPath =
        fs::current_path() /
        outDir / 
        ("keypoints_" + std::to_string(i) + ".jpg");

        bool success = cv::imwrite(outputPath.string(), out);

        std::cout << "Writing to: " << outputPath << '\n';
        std::cout << "Success: " << success << '\n';
    }
}
// Matching serquential images
void SFM::pairMatchingPoints(){
    // BFMatcher with default params
    cv::Ptr<cv::BFMatcher> bf = cv::BFMatcher::create(cv::NORM_L2, false);

    const fs::path outDir = fs::current_path() / "outputs" / "matches";
    fs::create_directories(outDir);

    matches.assign(descriptors.size() - 1, {});

    for (int i=0; i < descriptors.size() - 1; ++i){

        // Declare knn_matches for the given pair to be populated and filtered later
        std::vector<std::vector<cv::DMatch>> knn_matches;
        std::vector<cv::DMatch> good_matches;

        auto start = std::chrono::high_resolution_clock::now();
        
        bf -> knnMatch(descriptors[i], descriptors[i+1], knn_matches, 2);

        // Apply ratio test
        const float ratio_thresh = 0.75f;
        for (int i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i].size() >= 2) {
                if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
                    good_matches.push_back(knn_matches[i][0]);
                }
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double, std::milli>(end - start);

        matches[i] = std::move(good_matches);
        std::cout << "BFMatching: " << elapsed.count()
          << " ms, matches: "
          << matches[i].size() << '\n';

        // Draw the matches between the two images
        cv::Mat img1 = cv::imread(imgFiles[i], cv::IMREAD_COLOR);
        cv::Mat img2 = cv::imread(imgFiles[i + 1], cv::IMREAD_COLOR);

        cv::Mat out;
        cv::drawMatches(img1, keypoints[i], img2, keypoints[i + 1], matches[i], out,
                        cv::Scalar::all(-1), cv::Scalar::all(-1), {},
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        fs::path outPath = outDir /
            ("matches_" + std::to_string(i) + "_" + std::to_string(i + 1) + ".jpg");

        bool success = !out.empty() && cv::imwrite(outPath.string(), out);

        std::cout << "Writing to: " << outPath << '\n';
        std::cout << "Success: " << success << '\n';
    }
}

void SFM::loadCameras(){
    std::map<std::string, cv::Matx44d> mats;
    std::ifstream in(camerasPath);
    std::string line;

    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string key;
        ss >> key;

        cv::Matx44d M;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                ss >> M(r, c);

        mats[key] = M;
    }
    cameraMatrices = std::move(mats);
}

void SFM::triangulateMatches(){
    loadCameras();
    // Build projection matrices P_i = (world_mat_i * scale_mat_i)[:3,:4] for every image
    std::vector<cv::Matx34d> projMats(imgFiles.size());
    for (size_t k = 0; k < imgFiles.size(); ++k) {
        cv::Matx44d worldMat = cameraMatrices["world_mat_" + std::to_string(k)];
        cv::Matx44d scaleMat = cameraMatrices["scale_mat_" + std::to_string(k)];
        cv::Matx44d P4 = worldMat * scaleMat;

        cv::Matx34d P;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c)
                P(r, c) = P4(r, c);
        projMats[k] = P;
    }

    points3D.assign(matches.size(), {});

    for (int i = 0; i < matches.size(); i++){
        auto& currMatches = matches[i];
        if (currMatches.empty()) continue;

        std::vector<cv::Point2d> pts1, pts2;
        pts1.reserve(currMatches.size());
        pts2.reserve(currMatches.size());

        for (int j = 0; j < currMatches.size(); j++){
            auto sampleMatch = currMatches[j];
            int m = sampleMatch.queryIdx;
            int n = sampleMatch.trainIdx;
            cv::Point2f p1 = keypoints[i][m].pt;
            cv::Point2f p2 = keypoints[i + 1][n].pt;
            pts1.emplace_back(p1.x, p1.y);
            pts2.emplace_back(p2.x, p2.y);
        }

        cv::Mat X4; // 4 x N homogeneous points (3d)
        cv::triangulatePoints(cv::Mat(projMats[i]), cv::Mat(projMats[i + 1]), pts1, pts2, X4);

        std::vector<cv::Point3d> pts3d;
        pts3d.reserve(X4.cols);
        for (int col = 0; col < X4.cols; ++col) {
            double w = X4.at<double>(3, col);
            if (std::abs(w) < 1e-12) continue;
            pts3d.emplace_back(
                X4.at<double>(0, col) / w,
                X4.at<double>(1, col) / w,
                X4.at<double>(2, col) / w
            );
        }

        points3D[i] = std::move(pts3d);
        std::cout << "Triangulated " << points3D[i].size()
                  << " points for pair " << i << "->" << i + 1 << '\n';
    }
}

void SFM::exportPointsCSV() {
    std::string path = "check.csv";
    std::ofstream out(path);
    out << "x,y,z\n";
    for (const auto& pts : points3D)
        for (const auto& p : pts)
            out << p.x << "," << p.y << "," << p.z << "\n";
    std::cout << "Wrote CSV to " << path << '\n';
}