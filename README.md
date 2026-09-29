First run: 
```
$VCPKG_ROOT/vcpkg install --x-manifest-root=.
```
to download and compile the necessary libraries using vcpkg (I'm assuming vcpkg is installed already).

Then run:
```
cmake --preset vcpkg
cmake --build build
```
to build the executables.