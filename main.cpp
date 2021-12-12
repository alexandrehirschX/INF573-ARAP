#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject.h>
#include <vector>
#include <math.h>
#include <iostream>
#include<Eigen/Dense>
#include<fstream>
#include<Eigen/SparseCholesky>


using namespace Eigen;


MatrixXd Vo;
MatrixXd Vd, Vd2;

MatrixXi Fo;


float calc_cotangent(Vector3d v1, Vector3d v2){
    return 1/tan(acos(v1.normalized().dot(v2.normalized())));
}


bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
  if (key == '1')
  {
    viewer.data().clear();
    viewer.data().set_mesh(Vo, Fo);
    viewer.core().align_camera_center(Vo,Fo);
  }
  return false;
}

class Shape {
  public:
    MatrixXd V1, V2, W, L;
    MatrixXi F;
    std::vector<std::vector<int>> N;
    std::vector<Matrix3d> R;
    std::vector<int> Fixed;
    int n_vertices, n_fixed, n_faces;

    Shape(MatrixXd Vo, MatrixXi Fo){
      V1 = Vo;
      F = Fo;
      n_vertices = V1.rows();
      n_faces = F.rows();
      get_neighbors();
      get_weights();
      R.resize(n_vertices);
    }

    void deform(MatrixXd Vd, std::vector<int> Fi){
      V2 = Vd;
      n_fixed = Fi.size();
      Fixed.resize(n_fixed);
      Fixed = Fi;
    }

    void get_neighbors(){
      N.resize(n_vertices);
      std::vector<int> tmp;
      for(int i = 0; i < n_faces; i++){
        for(int j = 0; j < 3; j++){
          tmp = N[F(i,j)];
          for(int k = 0; k < 3; k++){
            if(k != j){
              if(std::find(tmp.begin(), tmp.end(), F(i,k)) == tmp.end()){
                N[F(i,j)].push_back(F(i,k));
              }
            }
          }
        }
      }
    }

    void get_weights(){
      W = MatrixXd::Zero(n_vertices, n_vertices);
      int j;
      float w;
      for(int i = 0; i < n_faces; i++){
        for(int n = 0; n < N[i].size(); n++){
          j = N[i][n];
          for(int k = 0; k < N[i].size(); k++){
            for(int l = 0; l < N[j].size(); l++){
              if (N[i][k] == N[j][l] && i < j){
                w = 0.5 * calc_cotangent(V1.row(i)-V1.row(N[i][k]), V1.row(j)-V1.row(N[i][k]));
                W(i,j) += w;
                W(j,i) += w;
              }
            }
          }
        }
      }
    }

    void get_rotations(){
      Matrix3d S, U, V;
      int in, j;
      for(int i = 0; i < n_vertices; i++){
        S.setZero();
        for(int n = 0; n < N[i].size(); n++){
          j = N[i][n];
          S += W(i,j) * (V1.row(i) - V1.row(j)).transpose() * (V2.row(i) - V2.row(j));
        }
        JacobiSVD<MatrixXd> svd(S, ComputeThinU | ComputeThinV);
        V = svd.matrixV(); 
        U = svd.matrixU();
        R[i] = V*U.transpose();
        if(R[i].determinant() <= 0){
          svd.singularValues().minCoeff(&in);
          U.col(in) *= -1;
          R[i] = V*U.transpose();
        }
      }
    }

    void get_p(){
      MatrixXd b = MatrixXd::Zero(n_vertices + n_fixed, 3);
      int j;
      for(int i = 0; i < n_vertices; i++){
        for(int n = 0; n < N[i].size(); n++){
          j = N[i][n];
          b.row(i) += W(i,j) * (R[i] + R[j]) *(V1.row(i) - V1.row(j)).transpose() * 0.5;
        }
      }
      for(int i = 0; i < n_fixed; i++){
        b.row(n_vertices + Fixed[i]) = V2.row(Fixed[i]);
      }
      b = L.inverse() * b;
      V2 = b.block(0, 0, n_vertices, 3);
    }

    MatrixXd ARAP(int n_iter){
      for(int iter = 0; iter < n_iter; iter++){
        get_rotations();
        L = MatrixXd::Zero(n_vertices + n_fixed, n_vertices + n_fixed);
        L.block(0, 0, n_vertices, n_vertices) = -W;
        L.block(0, 0, n_vertices, n_vertices).diagonal() = W.rowwise().sum();
        int I;
        for(int i = 0; i < n_fixed; i++){
          I = Fixed[i];
          L(I, n_vertices + I) = 1;
          L(n_vertices + I, I) = 1;
        }
        get_p();
      }
      return V2;
    }

};



int main(int argc, char *argv[])
{
    // Load a mesh in OFF format
    igl::readOFF("../meshes/bar1.off", Vo, Fo);
    std::cout << Fo.rows() << std::endl;
    // Plot the mesh

    Vd = Vo;
    for(int i = 49; i < 98; i++){
      Vd(i,0) += 50;
    }

    // Vd2 = Vo;
    // for(int i = 49; i < 98; i++){
    //   Vd2(i,0) -= 50;
    // }

    std::vector<int> Fixed(98);
    for(int i = 0; i < 98; i++){
      Fixed[i] = i;
    }

    Shape M(Vo, Fo);
    M.deform(Vd, Fixed);
    Vd = M.ARAP(5);

    //M.deform(Vd2, Fixed);
    //Vd = M.ARAP(5);

    igl::opengl::glfw::Viewer viewer;
    //viewer.callback_key_down = &key_down;
    viewer.data().set_mesh(Vd, Fo);
    viewer.launch();
}