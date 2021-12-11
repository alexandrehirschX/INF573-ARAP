#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>

#include <vector>
#include <math.h>

using namespace Eigen;

float calc_cotangent(Vector3d v1, Vector3d v2){
    return 1/tan(acos(v1.normalized().dot(v2.normalized())));
}

void weights(MatrixXd &V, MatrixXi &F, MatrixXd &W){
    float wij;
    for(int i = 0; i < V.rows(); i++){
        for(int j = 0; j < V.rows(); j++){
            wij = 0;
            for(int k = 0; k < F.rows(); k++){
                if(F(k,0) == i && F(k,1) == j || F(k,0) == j && F(k,1) == i){
                    wij += calc_cotangent(V.row(i)-V.row(F(k,2)), V.row(j)-V.row(F(k,2)));
                }
                else if(F(k,0) == i && F(k,2) == j || F(k,0) == j && F(k,2) == i){
                    wij += calc_cotangent(V.row(i)-V.row(F(k,1)), V.row(j)-V.row(F(k,1)));
                }
                else if(F(k,1) == i && F(k,2) == j || F(k,1) == j && F(k,2) == i){
                    wij += calc_cotangent(V.row(i)-V.row(F(k,0)), V.row(j)-V.row(F(k,0)));
                }
            }
            W(i,j) = wij/2;
        }
    }
}