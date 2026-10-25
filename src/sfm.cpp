#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <filesystem>
#include "sfm.h"

// Move keypoints and descriptors inside the class
// Create class
// Change iterator to first read all and then sort 
// Check why it takes so long 
namespace fs = std::filesystem;

SFM::SFM(std::string imgFolder, std::string maskFolder) : imgFolder(imgFolder), maskFolder(maskFolder){};

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

    for (int i=0; i < descriptors.size(); ++i){

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