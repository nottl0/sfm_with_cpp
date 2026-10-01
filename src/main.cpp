#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <iostream>

using namespace cv;
using namespace std;

int main() {

    // Read image (BGR color by default)
    cv::Mat img = cv::imread("/Users/aminaizbassar/Documents/cpp_proj/scan65/image/000000.png", cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Could not read image\n";
        return 1;
    }

    std::cout << "Size: " << img.cols << "x" << img.rows
              << ", channels: " << img.channels() << "\n";

    // int y = 50, x = 100;  // row, column (note: row first with .at)

    // Color pixel: Vec3b = 3 unsigned bytes, in B, G, R order
    // cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);
    // int b = pixel[0], g = pixel[1], r = pixel[2];
    // std::cout << "BGR: " << b << ", " << g << ", " << r << "\n";

    cv::Mat mask = cv::imread("/Users/aminaizbassar/Documents/cpp_proj/scan65/mask/000000.png", cv::IMREAD_GRAYSCALE);
    // sift doesn't have a constructer and can only be created with a create method
    // which returns a pointer
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    // create vector and matrix to store keypoints and descriptors per image
    // N: kp.pt (x, y position), kp.size, kp.angle, kp.response, kp.octave
    std::vector<cv::KeyPoint> keypoints;
    // N x 128, CV_32F
    cv::Mat descriptors;
    
    // const _OutputArray& wraps descriptors when its passed as an argument
    // const because in cpp we can't get a reference to a temporaty object 
    // keypoints are also passed by reference such that both are editable
    sift -> detectAndCompute(img, mask, keypoints, descriptors);

    std::cout << keypoints.size() << " keypoints, descriptors: "
              << descriptors.rows << "x" << descriptors.cols << "\n";

    // Visualize
    cv::Mat out;
    cv::drawKeypoints(img, keypoints, out, cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imwrite("keypoints.jpg", out);

    return 0;
}