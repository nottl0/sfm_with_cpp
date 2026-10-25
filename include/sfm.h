#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

// Can add optionality on the parameters off point matching or descriptor extraction methods
class SFM{

    std::string imgFolder;
    std::string maskFolder;
    std::vector<std::vector<cv::KeyPoint>> keypoints;
    std::vector<cv::Mat> descriptors;
    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<std::string> imgFiles;
    
    public:
        SFM(std::string imgFolder, std::string maskFolder);
        void computeSiftDescriptors();
        void pairMatchingPoints();
};