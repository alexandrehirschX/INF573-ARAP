#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject.h>
#include <igl/unproject_onto_mesh.h>
#include <vector>
#include <math.h>
#include <iostream>
#include<Eigen/Dense>
#include<fstream>
#include<Eigen/SparseCholesky>


using namespace Eigen;

typedef Triplet<double> T;
typedef SimplicialCholesky<SparseMatrix<double>> SC;

MatrixXd Vo;
MatrixXd Vd, Vd2;

MatrixXi Fo;


struct State
{
  // Rest and transformed control points
  Eigen::MatrixXd CV, CU;
  bool placing_handles = true;
} s;

float calc_cotangent(Vector3d v1, Vector3d v2){
    return 1/tan(acos(v1.normalized().dot(v2.normalized())));
}




class Shape {
  public:
    MatrixXd V1, V2, W, L, b;
    MatrixXi F;
    SparseMatrix<double> Z;
    std::vector<std::vector<int>> N;
    std::vector<Matrix3d> R;
    std::vector<int> Fixed;
    std::vector<MatrixXd> V1_diff, w_V1_diff, V2_diff;
    int n_vertices, n_fixed, n_faces;
    float current_energy;

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
      current_energy = 0;
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
      for(int i = 0; i < n_vertices; i++){
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

        std::vector<T> tripletList;
        //tripletList.reserve((L > 0).count());
        for(int i = 0; i < n_vertices + n_fixed; i++){
          for(int j = 0; j < n_vertices + n_fixed; j++){
            if (L(i,j) != 0){
              tripletList.push_back(T(i,j,L(i,j)));
            }
          }
        }

        Z.resize(n_vertices + n_fixed, n_vertices + n_fixed);
        Z.setFromTriplets(tripletList.begin(), tripletList.end());
      

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
      current_energy = 0;
      for(int i = 0; i < n_vertices; i++){
        S.setZero();
        for(int n = 0; n < N[i].size(); n++){
          j = N[i][n];
          S += w_V1_diff[i].col(j) * V2_diff[i].col(j).transpose();
          current_energy += W(i,j) * pow((V2_diff[i].col(j) - (R[i] * V1_diff[i].col(j))).norm(),2);
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

      SC chol(Z);
      //V2 = (L.inverse()*b).block(0, 0, n_vertices, 3);
      V2 = (chol.solve(b)).block(0, 0, n_vertices, 3);
    }


    MatrixXd ARAP(int n_iter){
      //std::cout << current_energy() << std::endl;
      float old_energy;
      for(int iter = 0; iter < n_iter; iter++){
        V2_diff = get_diff(V2);
        old_energy = current_energy;
        get_rotations();
        get_p();
        std::cout << current_energy << std::endl;
        if(abs(old_energy - current_energy) < 0.1){
          std::cout << "Iterations required: " << iter << std::endl;
          break;
        }
      }
      return V2;
    }

};


// bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
// {
//   switch(key){
//     case '1':
//     {

//     }
//     case '2':
//     {

      
//     }

//     default:
//       return false;

//   }
//   viewer.data().clear();
//   viewer.data().set_mesh(Vo, Fo);
//   viewer.core().align_camera_center(Vo,Fo);
//   return true;
// }


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

    // std::vector<int> Fixed(98);
    // for(int i = 0; i < 98; i++){
    //   Fixed[i] = i;
    // }

    // Shape M(Vo, Fo);
    // M.deform(Vd, Fixed);
    // Vd = M.ARAP(400);
    
    
    igl::opengl::glfw::Viewer viewer;

    viewer.callback_mouse_down = 
    [&](igl::opengl::glfw::Viewer&, int, int)->bool
  {
    RowVector3f last_mouse = Eigen::RowVector3f(
      viewer.current_mouse_x,viewer.core().viewport(3)-viewer.current_mouse_y,0);

      // Find closest point on mesh to mouse position
      int fid;
      Eigen::Vector3f bary;
      if(igl::unproject_onto_mesh(
        last_mouse.head(2),
        viewer.core().view,
        viewer.core().proj, 
        viewer.core().viewport, 
        Vo, Fo, 
        fid, bary))
      {
        long c;
        bary.maxCoeff(&c);
        Eigen::RowVector3d new_c = Vo.row(Fo(fid,c));
        std::cout << Fo(fid,c) << std::endl;
      }

    return false;
  };

    //M.deform(Vd2, Fixed);
    //Vd = M.ARAP(10);

    
    //viewer.callback_key_down = &key_down;
    viewer.data().set_mesh(Vo, Fo);
    viewer.launch();
}