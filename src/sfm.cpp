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

        std::cout << "SIFT: " << elapsed.count() << " ms, keypoints: " << keypoints[i].size() << '\n';

        cv::Mat out;
        cv::drawKeypoints(img, keypoints[i], out, cv::Scalar::all(-1),
                        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        const fs::path outDir = fs::current_path() / "outputs" / "keypoints";
        fs::create_directories(outDir);

        fs::path outputPath = fs::current_path() / outDir / ("keypoints_" + std::to_string(i) + ".jpg");

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
        std::cout << "BFMatching: " << elapsed.count() << " ms, matches: " << matches[i].size() << '\n';

        // Draw the matches between the two images
        cv::Mat img1 = cv::imread(imgFiles[i], cv::IMREAD_COLOR);
        cv::Mat img2 = cv::imread(imgFiles[i + 1], cv::IMREAD_COLOR);

        cv::Mat out;
        cv::drawMatches(img1, keypoints[i], img2, keypoints[i + 1], matches[i], out,
                        cv::Scalar::all(-1), cv::Scalar::all(-1), {},
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        fs::path outPath = outDir / ("matches_" + std::to_string(i) + "_" + std::to_string(i + 1) + ".jpg");

        bool success = !out.empty() && cv::imwrite(outPath.string(), out);

        std::cout << "Writing to: " << outPath << '\n';
        std::cout << "Success: " << success << '\n';
    }
}

// Returns K (normalized), R, t such that P ~ K [R | t]
static void decomposeP(const cv::Matx34d& P, cv::Matx33d& K, cv::Matx33d& R, cv::Vec3d& t) {
    cv::Mat Km, Rm, Ch;                       // Ch: homogeneous camera center (4x1)
    cv::decomposeProjectionMatrix(cv::Mat(P), Km, Rm, Ch);

    K = cv::Matx33d(Km.ptr<double>());        
    K = K * (1.0 / K(2, 2));
    R = cv::Matx33d(Rm.ptr<double>());

    cv::Vec3d C(Ch.at<double>(0) / Ch.at<double>(3),
                Ch.at<double>(1) / Ch.at<double>(3),
                Ch.at<double>(2) / Ch.at<double>(3));
    t = -(R * C);
}

void SFM::filterMatches(){
    const fs::path outDir = fs::current_path() / "outputs" / "filteredMatches";
    fs::create_directories(outDir);

    // We will try to get E (epipolar matrix) from the intrinsic matrix K and extrinsic matrix [R|t] because they are known
    // X_1 (camera 1 coords of point X_w) = R_1 * X_w + t_1
    // X_2 (camera 2 coords of point X_w) = R_2 * X_w + t_2
    // X_w = R_1^(-1) * [X_1 - t_1] -> X_w = R_1^(T) * [X_1 - t_1]
    // X_2 = R_2 * R_1^T * [X_1 - t_1] + t_2 -> X_2 = R_2 * R_1^T * X_1 + (t_2 - R_2 * R_1^T * t_1)
    // Knowing that for \lambda * x_2 = R \lambda * x_1 + t -> E = [t]_x * R 
    // E = [t_2 - R_2 * R_1^T * t_1]_x * R_2 * R_1^T

    filteredMatches.assign(matches.size(), std::vector<cv::DMatch>{});

    for (size_t i = 0; i < matches.size(); ++i) {
        // Get the world matrices for a pair of
        cv::Matx44d W1 =
            cameraMatrices.at("world_mat_" + std::to_string(i));
        cv::Matx44d W2 =
            cameraMatrices.at("world_mat_" + std::to_string(i + 1));

        // Discard the last row
        cv::Matx34d P1, P2;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                P1(r, c) = W1(r, c);
                P2(r, c) = W2(r, c);
            }
        }

        cv::Matx33d K, K2, R1, R2;
        cv::Vec3d t1, t2;
        decomposeP(P1, K,  R1, t1);
        decomposeP(P2, K2, R2, t2);      // K2 should equal K; only used for the sanity check

        cv::Matx33d K_inv = K.inv();
        cv::Matx33d R = R2 * R1.t();
        cv::Vec3d t = t2 - R * t1;

        cv::Matx33d tx(
             0,    -t(2),  t(1),
             t(2),  0,    -t(0),
            -t(1),  t(0),  0
        );
        // Compute final Essential matrix for the pair
        cv::Matx33d E = tx * R;

        // Filter points
        auto& currMatches = matches[i];
        if (currMatches.empty()) continue;

        for (int j = 0; j < currMatches.size(); j++){
            auto sampleMatch = currMatches[j];
            int m = sampleMatch.queryIdx;
            int n = sampleMatch.trainIdx;
            cv::Point2f p1 = keypoints[i][m].pt;
            cv::Point2f p2 = keypoints[i + 1][n].pt;

            cv::Vec3d p1_hg(p1.x, p1.y, 1);
            cv::Vec3d p2_hg(p2.x, p2.y, 1);

            // Get the coordinates of matched points in camera frame from image 1 and 2
            cv::Vec3d X1 = K_inv * p1_hg;
            cv::Vec3d X2 = K_inv * p2_hg;

            // Epipolar line
            cv::Vec3d line = E * X1;
            
            // dist = 0 when X2 lies on the epipolar line, to have meaningful distance normalize by a, b coef in aX1[0] + bX1[1] + c = 0 (epipolar line)  
            double denom = std::sqrt(line[0]*line[0] + line[1]*line[1]);

            // Check for 0 division
            if (denom < 1e-12) continue;
            double dist = std::abs(X2.dot(line)) / denom;

            // Started with random
            const double pixelThresh = 1.0;                    
            const double normThresh  = pixelThresh * K_inv(0, 0);

            if (dist < normThresh){
                filteredMatches[i].emplace_back(sampleMatch);
            }
        }          

        cv::Mat img1 = cv::imread(imgFiles[i], cv::IMREAD_COLOR);
        cv::Mat img2 = cv::imread(imgFiles[i + 1], cv::IMREAD_COLOR);

        cv::Mat out;
        cv::drawMatches(img1, keypoints[i], img2, keypoints[i + 1], filteredMatches[i], out,
                        cv::Scalar::all(-1), cv::Scalar::all(-1), {},
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        fs::path outPath = outDir / ("matches_" + std::to_string(i) + "_" + std::to_string(i + 1) + ".jpg");

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
    // The dataset first scales the world coordinates to fit the object into a unit circle, then transforms from world coordinates to pixel coordinates
    std::vector<cv::Matx34d> projMats(imgFiles.size());
    for (size_t k = 0; k < imgFiles.size(); ++k) {
        cv::Matx44d worldMat = cameraMatrices["world_mat_" + std::to_string(k)];
        cv::Matx44d scaleMat = cameraMatrices["scale_mat_" + std::to_string(k)];
        // Because the original scene coordinates were scaled before projecting them to camera coordinates multiply the world matrix with scale matrix
        // The matrices are 4 by 4 with the last row being [0 0 0 1]
        cv::Matx44d P4 = worldMat * scaleMat;

        // Populate the corresponding entry in projMats
        cv::Matx34d P;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c)
                P(r, c) = P4(r, c);
        projMats[k] = P;
    }

    filterMatches();
    points3D.assign(matches.size(), {});

    std::cout << "Len matches 0: " << matches[0].size();
    std::cout << "Len filtered matches 0: " << filteredMatches[0].size();

    // For each image pair
    for (int i = 0; i < filteredMatches.size(); i++){
        auto& currMatches = filteredMatches[i];
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
    
        // Triangulate points from 2 images using their pixel coordinates
        cv::Mat X4;
        cv::triangulatePoints(cv::Mat(projMats[i]), cv::Mat(projMats[i + 1]), pts1, pts2, X4);
    
        // The 3d homogenious coordinates need to be normalized by the value in the 4th dimension 
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