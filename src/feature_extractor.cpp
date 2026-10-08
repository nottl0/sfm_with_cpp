#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <filesystem>

// Move keypoints and descriptors inside the class
// Create class
// Change iterator to first read all and then sort 
// Check why it takes so long 
namespace fs = std::filesystem;

void getSiftDescriptors(std::string imgFolder, std::string maskFolder, std::vector<std::vector<cv::KeyPoint>>& keypoints, std::vector<cv::Mat>& descriptors){
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

    // For each ffile in the folder collect keypoints and descriptors
    // Use sorted filenames
    for (int i = 0; i < imgPaths.size(); ++i) {
        cv::Mat img = cv::imread(imgPaths[i], cv::IMREAD_COLOR);
        cv::Mat mask = cv::imread(maskPaths[i], cv::IMREAD_GRAYSCALE);
        
        keypoints.emplace_back();
        descriptors.emplace_back();

        auto start = std::chrono::high_resolution_clock::now();

        sift -> detectAndCompute(img, mask, keypoints.back(), descriptors.back());

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed =
        std::chrono::duration<double, std::milli>(end - start);

        std::cout << "SIFT: " << elapsed.count()
          << " ms, keypoints: "
          << keypoints.back().size() << '\n';

        cv::Mat out;
        cv::drawKeypoints(img, keypoints[i], out, cv::Scalar::all(-1),
                        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        fs::path outputPath =
        fs::current_path() /
        "outputs" /
        ("keypoints_" + std::to_string(i) + ".jpg");

        bool success = cv::imwrite(outputPath.string(), out);

        std::cout << "Writing to: " << outputPath << '\n';
        std::cout << "Success: " << success << '\n';
        // std::string out_name = "./outputs/keypoints_" + std::to_string(i) + ".jpg";
        // cv::imwrite(out_name, out);
    }
}