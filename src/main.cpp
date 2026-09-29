#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <iostream>

// Simple Ceres Cost Function: f(x) = (10 - x)^2
struct CostFunctor {
    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        residual[0] = 10.0 - x[0];
        return true;
    }
};

int main() {
    std::cout << "--- Testing Included Packages ---\n\n";

    // ==========================================
    // 1. OpenCV Example
    // ==========================================
    // Create a 300x300 blue image (BGR format)
    cv::Mat image(300, 300, CV_8UC3, cv::Scalar(255, 0, 0));
    std::cout << "[OpenCV] Created matrix of size: " 
              << image.cols << "x" << image.rows << " pixels.\n";

    // ==========================================
    // 2. Eigen Example
    // ==========================================
    // Create a 3x3 rotation/transformation matrix
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    Eigen::Vector3d translation(1.0, 2.0, 3.0);
    std::cout << "[Eigen] Translation vector:\n" << translation.transpose() << "\n";

    // ==========================================
    // 3. Ceres Solver Example
    // ==========================================
    double x = 0.0; // Initial guess for optimization
    ceres::Problem problem;

    // Add cost function residual block
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor),
        nullptr,
        &x
    );

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false; // Set to true to see solver steps

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << "[Ceres] Optimized value x = " << x << " (Expected: 10.0)\n\n";

    std::cout << "All packages successfully loaded and executed!\n";
    return 0;
}