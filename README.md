## The project

This is a refresher project I did to practice C++ and Structure from Motion (SfM). It is inspired by [minecraft_slam](https://github.com/Luizerko/minecraft_slam/tree/main/sfm), and it is intended as a small, readable example rather than a general-purpose reconstruction library.

The program uses known camera projection matrices and SIFT features to reconstruct a sparse 3D point cloud from a sequence of overlapping images. The main processing steps are:

1. Read the object images and their masks.
2. Detect SIFT keypoints and descriptors.
3. Match consecutive image pairs with a brute-force matcher.
4. Filter the matches using the epipolar constraint derived from the known camera matrices.
5. Triangulate the remaining correspondences.
6. Export the 3D points to a CSV file.
7. Optionally visualize the point cloud with Python.

- `scan65/`: contains the example input from [RoboImagedata](https://roboimagedata.compute.dtu.dk/). It includes an object photographed from different viewpoints, masks for the images, and the camera calibration matrices. The masks isolate the foreground object so that SIFT is not distracted by the background.

  - `scan65/image/` contains the input images.
  - `scan65/mask/` contains the corresponding binary masks.
  - `scan65/cameras.npz` contains the original camera matrices as NumPy arrays.
  - `scan65/cameras.txt` is a plain-text version of `cameras.npz`, generated for convenient loading from C++.

- `src/main.cpp`: defines the application entry point and runs the SfM pipeline.
- `src/sfm.cpp`: implements SIFT extraction, matching, epipolar filtering, camera loading, triangulation, and CSV export.
- `include/sfm.h`: declares the `SFM` class used by the project.
- `python_code/`: contains utilities for converting and visualizing the data.
  - `transform_npz_to_txt.py` converts `cameras.npz` into `cameras.txt`.
  - `visualize_points.py` centers and renders `check.csv` as a 3D point cloud.
  - `run_sift.py` provides a separate Python SIFT example.
- `outputs/`: receives generated diagnostic images.
- `check.csv`: is the generated sparse point cloud.

## Pipeline

### 1. Detect features

`SFM::computeSiftDescriptors()` reads the images in alphabetical order, applies their matching masks, and runs OpenCV SIFT. The keypoints and descriptors for every image are stored in memory. A visualization is written for each image in `outputs/keypoints/`.

Image and mask filenames are paired by name. The mask lookup in `sfm.cpp` is based on a filename substring, so the provided `scan65/` layout should be preserved when replacing the example data.

### 2. Match consecutive views

`SFM::pairMatchingPoints()` compares each image to the next image in sequence with `cv::BFMatcher` using the L2 norm. It uses OpenCV's two-nearest-neighbor ratio test with a threshold of `0.75`. The resulting pairwise matches are written to `outputs/matches/`.

This is intentionally a sequential matcher: view $i$ is matched only with view $i+1$. A denser reconstruction would match more overlapping views or add loop-closing matches.

### 3. Filter with epipolar geometry

`SFM::triangulateMatches()` loads the camera matrices, builds projection matrices, and calls `SFM::filterMatches()`. For each consecutive pair, the code derives the relative rotation and translation from the known world matrices. It then computes the essential matrix

$$
E = [t]_\times R
$$

and discards a match when its point is farther than one pixel from the corresponding epipolar line.

Using known camera poses makes this reconstruction a posed SfM example. Camera motion estimation and bundle adjustment are not performed here.

### 4. Triangulate points

For every accepted match, `SFM::triangulateMatches()` combines the two camera projection matrices with OpenCV's `cv::triangulatePoints()`. Homogeneous 3D coordinates are normalized by their fourth component and stored in `SFM::points3D`.

The current implementation triangulates only the sequential pairs used during matching. Points seen in more than two views are not globally merged or deduplicated.

### 5. Export the sparse cloud

`SFM::exportPointsCSV()` writes the reconstructed points to `check.csv` in this format:

```text
x,y,z
0.123456, -0.789012, 1.345678
...
```

The coordinates are in the scene coordinate system defined by the supplied matrices.

## Dependencies

The C++ build uses:

- OpenCV — SIFT, keypoint matching, drawing, image I/O, and triangulation
- Ceres Solver — linked by the project and available for future bundle adjustment
- Eigen — matrix and vector utilities
- C++17 filesystem support

Python visualization uses NumPy, pandas, and matplotlib. It requires the input CSV and can display interactively on macOS, for example with the built-in macOS backend:

```bash
python3 -m pip install numpy pandas matplotlib
```

## Prerequisites

- [CMake](https://cmake.org/) 3.25 or newer
- [Ninja](https://ninja-build.org/)
- [vcpkg](https://github.com/microsoft/vcpkg), cloned and bootstrapped
- A C++17 compiler (Xcode command line tools on macOS)

Set `VCPKG_ROOT` to your vcpkg directory. Add this to `~/.zshrc`:

```bash
export VCPKG_ROOT=/path/to/vcpkg
```

Then open a new terminal.

## Build

Run these from the project root:

```bash
cmake --preset vcpkg
cmake --build build
```

The first configure downloads and builds the dependencies listed in `vcpkg.json`, including OpenCV. This can take a long time, but normally only needs to happen once. The executable is placed in `build/sfm_with_cpp`.

To force a clean rebuild after changing build configuration or dependencies:

```bash
rm -rf build
cmake --preset vcpkg
cmake --build build
```

## Run

The executable expects to be run from the project root because `src/main.cpp` uses relative paths:

```bash
./build/sfm_with_cpp
```

The program reports progress in the terminal, including timing for SIFT and matching, the number of accepted matches, triangulated point counts, and the CSV destination. It also generates:

- `outputs/keypoints/keypoints_<image>.jpg` — detected SIFT features
- `outputs/matches/matches_<i>_<i+1>.jpg` — raw ratio-tested matches
- `outputs/filteredMatches/matches_<i>_<i+1>.jpg` — matches retained by epipolar filtering
- `check.csv` — triangulated 3D coordinates

## Camera data

Each row in `scan65/cameras.txt` starts with a matrix name followed by 16 floating-point values. The file contains 4 × 4 matrices such as `world_mat_0`, `scale_mat_0`, and the corresponding matrices for subsequent views.

The matrices are paired by image index. `world_mat_i` changes the scene from world coordinates to camera coordinates, while `scale_mat_i` is applied to the world coordinates before projection. The C++ code uses:

$$
P_i = (W_i S_i)_{[:3,:4]},
$$

where $W_i$ is `world_mat_i`, $S_i$ is `scale_mat_i`, and $P_i$ is the 3 × 4 projection matrix used for triangulation.

For a different camera dataset, the matrices must use this same 4 × 4 text format and be correctly associated with the images. `cameras.npz` can be converted with:

```bash
cd python_code
python3 transform_npz_to_txt.py
```

The script is relative-path based, so it should be run from `python_code/` as shown above.

## Visualize the point cloud

After running the C++ executable, center and plot the generated points:

```bash
cd python_code
python3 visualize_points.py
```

The visualization script subtracts the mean of each coordinate axis and uses equal axis scaling. It does not modify `check.csv`.

## Expected workflow

```bash
# Configure and build
cmake --preset vcpkg
cmake --build build

# Run the reconstruction from the project root
./build/sfm_with_cpp

# Inspect the 3D output
cd python_code
python3 visualize_points.py
```

## Project limitations

This repository is a compact learning implementation. Its current scope is intentionally limited:

- only consecutive image pairs are matched;
- points are not globally tracked or deduplicated across multiple views;
- no global bundle adjustment or camera-pose optimization is performed;
- camera matrices are supplied rather than estimated;
- no robust false-match filtering such as RANSAC is used in matching;
- the epipolar tolerance is fixed at one normalized pixel;
- all input data is loaded from relative paths;
- running the executable writes directly into `check.csv` and the `outputs/` directory.
