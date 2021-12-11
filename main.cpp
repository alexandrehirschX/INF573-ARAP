#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <vector>
#include <math.h>
#include "weights.cpp"

using namespace Eigen;


MatrixXd V1;
MatrixXi F1;

void ARAP(MatrixXd &V, MatrixXi &F, VectorXd &D){
    for(int i = 0; i < D.rows(); i++){
        V(D(i), 0) += 0.05;
    }

    
}

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
  if (key == '1')
  {
    VectorXd D(2);
    D << 374, 382;
    ARAP(V1,F1,D);
    viewer.data().clear();
    viewer.data().set_mesh(V1, F1);
    viewer.core().align_camera_center(V1,F1);
  }
  return false;
}




int main(int argc, char *argv[])
{
    // Load a mesh in OFF format
    igl::readOFF("../meshes/cactus_small.off", V1, F1);

    // Plot the mesh


    int n = V1.rows();
    MatrixXd W1(n,n);

    //weights(V1, F1, W1);
    
    
    igl::opengl::glfw::Viewer viewer;

    //viewer.callback_key_down = &key_down;
    viewer.data().set_mesh(V1, F1);
    viewer.launch();
}