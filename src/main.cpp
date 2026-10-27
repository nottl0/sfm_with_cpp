#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <iostream>
#include <filesystem>
#include "sfm.h"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

int main() {
    std::cout << fs::current_path() << '\n';
    std::string imgFolder = "./scan65/image";
    std::string maskFolder = "./scan65/mask";
    std::string camerasPath = "./scan65/cameras.txt";

    SFM sfm(imgFolder, maskFolder, camerasPath);
    sfm.computeSiftDescriptors();
    sfm.pairMatchingPoints();
    sfm.triangulateMatches();
    sfm.exportPointsCSV();
    return 0;
}