#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

void getSiftDescriptors(std::string imgFolder, std::string maskFolder, std::vector<std::vector<cv::KeyPoint>>& keypoints, std::vector<cv::Mat>& descriptors);