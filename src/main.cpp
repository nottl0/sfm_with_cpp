#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <iostream>
#include <filesystem>
#include "sfm.h"

// Next step is using descriptors and pairing them across adjasent images
using namespace cv;
using namespace std;
namespace fs = std::filesystem;

int main() {
    std::cout << fs::current_path() << '\n';
    std::string imgFolder = "./scan65/image";
    std::string maskFolder = "./scan65/mask";

    SFM sfm(imgFolder, maskFolder);
    sfm.computeSiftDescriptors();
    sfm.pairMatchingPoints();
    return 0;
}