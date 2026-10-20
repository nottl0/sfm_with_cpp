#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <iostream>
#include <filesystem>
#include "feature_extractor.h"


using namespace cv;
using namespace std;
namespace fs = std::filesystem;

int main() {
    std::cout << fs::current_path() << '\n';
    std::string imgFolder = "./scan65/image";
    std::string maskFolder = "./scan65/mask";

    std::vector<std::vector<cv::KeyPoint>> keypoints;
    std::vector<cv::Mat> descriptors;
    SiftExtractorWrapper siftWrapper(imgFolder, maskFolder, keypoints, descriptors);
    siftWrapper.getSiftDescriptors();

    return 0;
}