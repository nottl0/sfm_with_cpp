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

The first configure downloads and compiles all dependencies listed in `vcpkg.json`, including OpenCV. This can take a long time, but it only happens once. The executables are placed in `build/`.