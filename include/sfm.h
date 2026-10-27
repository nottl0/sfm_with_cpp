#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

// Can add optionality on the parameters off point matching or descriptor extraction methods
class SFM{

    std::string imgFolder;
    std::string maskFolder;
    std::string camerasPath;
    std::vector<std::vector<cv::KeyPoint>> keypoints;
    std::vector<cv::Mat> descriptors;
    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<std::vector<cv::DMatch>> filteredMatches;
    std::vector<std::string> imgFiles;
    std::map<std::string, cv::Matx44d> cameraMatrices;
    std::vector<std::vector<cv::Point3d>> points3D;

    public:
        SFM(std::string imgFolder, std::string maskFolder, std::string camerasPath);
        void computeSiftDescriptors();
        void pairMatchingPoints();
        void filterMatches();
        void triangulateMatches();
        void loadCameras();
        void exportPointsCSV();
};