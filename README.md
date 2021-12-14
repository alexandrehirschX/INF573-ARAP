# INF574: ARAP surface modelling implementation

To compile the code:
mkdir build
cd build
cmake ../
make

Run the code with:
./example_bin [OPTIONAL: mesh_file, default ../meshes/bar3.off] [OPTIONAL: is_animating (0,1), default: 0]

Examples:
./example_bin
./example_bin ../meshes/armadillo_1k.off
./example_bin ../meshes/armadillo_1k.off 1

