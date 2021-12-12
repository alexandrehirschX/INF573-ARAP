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
    MatrixXd V1, V2, W, L, b;
    MatrixXi F;
    std::vector<std::vector<int>> N;
    std::vector<Matrix3d> R;
    std::vector<int> Fixed;
    std::vector<MatrixXd> V1_diff, w_V1_diff, V2_diff;
    int n_vertices, n_fixed, n_faces;

    Shape(MatrixXd Vo, MatrixXi Fo){
      V1 = Vo;
      F = Fo;

      n_vertices = V1.rows();
      n_faces = F.rows();

      V1_diff = get_diff(V1);
      
      get_neighbors();
      get_weights();

      w_V1_diff.resize(n_vertices);
      for(int i = 0; i < n_vertices; i++){
        w_V1_diff[i] = MatrixXd::Zero(3, n_vertices);
        for(int j = 0; j < n_vertices; j++){
          w_V1_diff[i].col(j) = W(i,j) * V1_diff[i].col(j);
        }
      }

      R.resize(n_vertices);
    }

    void deform(MatrixXd Vd, std::vector<int> Fi){
      V2 = Vd;
      n_fixed = Fi.size();
      Fixed.resize(n_fixed);
      Fixed = Fi;
      get_laplacian_and_b();
    }

    std::vector<MatrixXd> get_diff(MatrixXd &V){
      std::vector<MatrixXd> D(n_vertices);
      for(int i = 0; i < n_vertices; i++){
        D[i] = MatrixXd::Zero(3, n_vertices);
        for(int j = 0; j < n_vertices; j++){
          D[i].col(j) = (V.row(i) - V.row(j)).transpose();
        }
      }
      return D;
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
                w = 0.5 * calc_cotangent(V1_diff[i].col(N[i][k]), V1_diff[j].col(N[i][k]));
                W(i,j) += w;
                W(j,i) += w;
              }
            }
          }
        }
      }
    }

    void get_laplacian_and_b(){
        L = MatrixXd::Zero(n_vertices + n_fixed, n_vertices + n_fixed);
        L.block(0, 0, n_vertices, n_vertices) = -W;
        L.block(0, 0, n_vertices, n_vertices).diagonal() = W.rowwise().sum();
        int I;
        for(int i = 0; i < n_fixed; i++){
          I = Fixed[i];
          L(I, n_vertices + I) = 1;
          L(n_vertices + I, I) = 1;
        }

        b = MatrixXd::Zero(n_vertices + n_fixed, 3);
        int j;
        for(int i = 0; i < n_vertices; i++){
          for(int n = 0; n < N[i].size(); n++){
            j = N[i][n];
            b.row(i) += 0.5 * (R[i] + R[j]) * w_V1_diff[i].col(j);
          }
        }
        for(int i = 0; i < n_fixed; i++){
          b.row(n_vertices + Fixed[i]) = V2.row(Fixed[i]);
        }

    }

    void get_rotations(){
      Matrix3d S, U, V;
      int in, j;
      for(int i = 0; i < n_vertices; i++){
        S.setZero();
        for(int n = 0; n < N[i].size(); n++){
          j = N[i][n];
          S += w_V1_diff[i].col(j) * V2_diff[i].col(j).transpose();
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
      b.block(0,0,n_vertices,3).setZero();
      int j;
      for(int i = 0; i < n_vertices; i++){
        for(int n = 0; n < N[i].size(); n++){
          j = N[i][n];
          b.row(i) += 0.5 * (R[i] + R[j]) * w_V1_diff[i].col(j);
        }
      }
      V2 = (L.llt().solve(b)).block(0, 0, n_vertices, 3);
    }


    float current_energy(){
      float S;
      Vector3d X;
      int j;
      for(int i = 0; i < n_vertices; i++){
        for(int n = 0; n < N[i].size(); n++){
          j = N[i][n];
          X = V2_diff[i].col(j) - (R[i] * V1_diff[i].col(j)); //(V2.row(i) - V2.row(j)).transpose()
          //std::cout << X.rows() << " " << X.cols() << std::endl;
          S += W(i,j) * pow(X.norm(),2);
        }
      }
      return S;
    }

    MatrixXd ARAP(int n_iter){
      //std::cout << current_energy() << std::endl;
      for(int iter = 0; iter < n_iter; iter++){
        V2_diff = get_diff(V2);
        std::cout << current_energy() << std::endl;
        get_rotations();
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
      Vd(i,0) += 100;
      Vd(i,1) -= 50; 
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
    Vd = M.ARAP(10);

    //M.deform(Vd2, Fixed);
    //Vd = M.ARAP(5);

    igl::opengl::glfw::Viewer viewer;
    //viewer.callback_key_down = &key_down;
    viewer.data().set_mesh(Vd, Fo);
    viewer.launch();
}