#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject.h>
#include <vector>
#include <math.h>
//#include "weights.cpp"
#include <iostream>
#include<Eigen/Dense>
#include<fstream>
// #include<Eigen/SparseCholesky>
// typedef Eigen::SparseMatrix<double> SpMat;
using namespace Eigen;


MatrixXd Vo;
MatrixXd Vd;

MatrixXi Fo;



void get_neighbors(MatrixXi &F, std::vector<std::vector<int>> &N){
  std::vector<int> tmp;
  for(int i = 0; i < F.rows(); i++){
    for(int j = 0; j < F.cols(); j++){
      tmp = N[F(i,j)];
      for(int k = 0; k < F.cols(); k++){
        if(k != j){
          if(std::find(tmp.begin(), tmp.end(), F(i,k)) == tmp.end()){
            N[F(i,j)].push_back(F(i,k));
          }
        }
      }
    }
  }
}

float calc_cotangent(Vector3d v1, Vector3d v2){
    return 1/tan(acos(v1.normalized().dot(v2.normalized())));
}

void get_weights(MatrixXd &V, MatrixXi &F, std::vector<std::vector<int>> &N, MatrixXd &W){
  W = MatrixXd::Zero(V.rows(), V.rows());
  int j;
  for(int i = 0; i < V.rows(); i++){
    for(int n = 0; n < N[i].size(); n++){
      j = N[i][n];
      for(int k = 0; k < N[i].size(); k++){
        for(int l = 0; l < N[j].size(); l++){
          if (N[i][k] == N[j][l]){
            W(i,j) += calc_cotangent(V.row(i)-V.row(N[i][k]), V.row(j)-V.row(N[i][k])) / 2;
          }
        }
      }
    }
  }
}

void get_rotations(MatrixXd &V1, MatrixXd &V2,std::vector<std::vector<int>> &N, MatrixXd &W, std::vector<Matrix3d> &R){
  Matrix3d S, U, V;
  int in, j;
  for(int i = 0; i < R.size(); i++){
    S = Matrix3d::Zero();
    for(int n = 0; n < N[i].size(); n++){
      j = N[i][n];
      S += W(i,j) * (V1.row(i) - V1.row(j)).transpose() * (V2.row(i) - V2.row(j));
    }
    //S.transposeInPlace();
    JacobiSVD<MatrixXd> svd(S, ComputeThinU | ComputeThinV);
    V = svd.matrixV().transpose();
    U = svd.matrixU();
    R[i] = V*U.transpose();
    if(R[i].determinant() <= 0){
      svd.singularValues().minCoeff(&in);
      U.col(in) *= -1;
      R[i] = V*U.transpose();
      //std::cout << R[i].determinant() << std::endl;
    }
    //R[i].transposeInPlace();
  }
}


void get_p(MatrixXd &V1, std::vector<Matrix3d> &R, std::vector<std::vector<int>> &N, MatrixXd &W, MatrixXd &L, MatrixXd &V2){
  
  Eigen::SparseMatrix<double> lb_operator_;
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_;
  
  MatrixXd b = MatrixXd::Zero(V1.rows() + 98, V1.cols());
  int n = V1.rows();
  int j;
  for(int i = 0; i < V1.rows(); i++){
    for(int n = 0; n < N[i].size(); n++){
      j = N[i][n];
      b.row(i) += W(i,j) * (R[i] + R[j]) *(V1.row(i) - V1.row(j)).transpose() / 2;
    }
  }
  for(int i = 0; i < 98; i++){
    b.row(n+i) = V1.row(i);
  }
  //b = L.colPivHouseholderQr().solve(b);
  b = L.inverse() * b;

  V2 = b.block(0,0, V1.rows(), 3);
  //V2.block(0,0,98,3) = Right.block(V1.rows(),0,98,3);
};




void ARAP(MatrixXd &V1, MatrixXd &V2, std::vector<std::vector<int>> &N, MatrixXd &W){

    for(int iter = 0; iter < 1; iter++){
      std::vector<Matrix3d> R(V1.rows());
      get_rotations(V1, V2, N, W, R);
      int n = W.rows();
      MatrixXd L = MatrixXd::Zero(n+98, n+98);
      L.block(0,0,n,n) = -W;
      //L = W;
      for(int i = 0; i < W.rows(); i++){
        L(i,i) = W.row(i).sum();
      }
      //L *= -1; 
      for(int i = 0; i < 98; i++){
        //L.row(i).setZero(); L.col(i).setZero();
        L(i, n+i) = 1;
        L(n+i, i) = 1;
      }
      get_p(V1, R, N, W, L, V2);

    }

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

int main(int argc, char *argv[])
{
    // Load a mesh in OFF format
    igl::readOFF("../meshes/bar1.off", Vo, Fo);

    // Plot the mesh

    Vd = Vo;
    for(int i = 49; i < 98; i++){
      Vd(i,0) += 10;
    }
    

    int n = Vo.rows(), m = Fo.rows();

    std::vector<std::vector<int>> N(m);
    get_neighbors(Fo, N);

    MatrixXd W(n,n);
    get_weights(Vo, Fo, N, W);

    // const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", "\n");
    // std::ofstream file("test.csv");
    // if (file.is_open())
    // {
    //     file << W.format(CSVFormat);
    //     file.close();
    // }
    ARAP(Vo, Vd, N, W);

    igl::opengl::glfw::Viewer viewer;

    //viewer.callback_key_down = &key_down;
    viewer.data().set_mesh(Vd, Fo);
    viewer.launch();
}