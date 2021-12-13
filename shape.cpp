
#include <vector>
#include <math.h>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/Core>


using namespace Eigen;

typedef Triplet<double> T;

float calc_cotangent(Vector3d v1, Vector3d v2){
    return 1/tan(acos(v1.normalized().dot(v2.normalized())));
}

class Shape {
  public:
    MatrixXd V1, V2, W, b;
    MatrixXi F;
    std::vector<std::vector<int>> N;
    std::vector<Matrix3d> R;
    VectorXi Fixed;
    std::vector<MatrixXd> V1_diff, w_V1_diff, V2_diff;
    int n_vertices, n_fixed, n_faces, to_reserve;
    float current_energy;
    SparseMatrix<double> L;
    std::vector<T> GlobaltripletList;
    SimplicialCholesky<SparseMatrix<double>> chol;

    Shape(MatrixXd Vo, MatrixXi Fo){
      V1 = Vo;
      F = Fo;

      n_vertices = V1.rows();
      n_faces = F.rows();


      V1_diff.resize(n_vertices);
      for(int i = 0; i < n_vertices; i++){
        V1_diff[i] = MatrixXd::Zero(3, n_vertices);
        for(int j = 0; j < n_vertices; j++){
          V1_diff[i].col(j) = (V1.row(i) - V1.row(j)).transpose();;
        }
      }

      get_neighbors();
      get_weights();


      w_V1_diff.resize(n_vertices);
      for(int i = 0; i < n_vertices; i++){
        w_V1_diff[i] = MatrixXd::Zero(3, n_vertices);
        for(int j = 0; j < n_vertices; j++){
          w_V1_diff[i].col(j) = W(i,j) * V1_diff[i].col(j);
        }
      }

      //Initializing rotations
      R.resize(n_vertices);
      Matrix3d S, U, V;
      int in,j;
      for(int i = 0; i < n_vertices; i++){
        S.setZero();
        for(int n = 0; n < N[i].size(); n++){
          j = N[i][n];
          S += w_V1_diff[i].col(j) * V1_diff[i].col(j).transpose(); //V2.row(i) - V2.row(j) V2_diff[i].col(j).transpose();
          //current_energy += W(i,j) * pow(( (v2i-v2j).transpose() - (R[i] * V1_diff[i].col(j))).norm(),2);
        }
        JacobiSVD<MatrixXd> svd(S, ComputeThinU | ComputeThinV);
        V = svd.matrixV(); U = svd.matrixU();
        R[i] = V*U.transpose();
        if(R[i].determinant() <= 0){
          svd.singularValues().minCoeff(&in);
          U.col(in) *= -1;
          R[i] = V*U.transpose();
        }
      }

    }

    void fix(VectorXi Fi){
      n_fixed = Fi.size();
      Fixed = Fi;
      //get_laplacian();
      std::vector<T> tripletList;
      tripletList.reserve(to_reserve + 2*n_fixed);
      
      int j;
      for(int i = 0; i < n_vertices; i++){
        for(int n = 0; n < N[i].size(); n++){
          j = N[i][n];
          tripletList.push_back(T(i,j, -W(i,j)));
        }
        tripletList.push_back(T(i,i, W.row(i).sum()));
      }

      int I, J;
      for(int i = 0; i < n_fixed; i++){
        I = Fixed[i]; J = n_vertices + i;
        tripletList.push_back(T(I, J, 1));
        tripletList.push_back(T(J, I, 1));
      }
      L.resize(n_vertices + n_fixed, n_vertices + n_fixed);
      L.setFromTriplets(tripletList.begin(), tripletList.end());
      chol.compute(L);
    }

    void deform(MatrixXd Vd){
      V2 = Vd;
      int n_deformed = V2.rows();
      //get_b();
      b = MatrixXd::Zero(n_vertices + n_fixed, 3);
      for(int i = 0; i < n_fixed; i++){
        b.row(n_vertices + i) = V2.row(i);
      }
      current_energy = 0;
    }

    void get_neighbors(){
      to_reserve = n_vertices;
      N.resize(n_vertices);
      std::vector<int> tmp;
      for(int i = 0; i < n_faces; i++){
        for(int j = 0; j < 3; j++){
          tmp = N[F(i,j)];
          for(int k = 0; k < 3; k++){
            if(k != j){
              if(std::find(tmp.begin(), tmp.end(), F(i,k)) == tmp.end()){
                N[F(i,j)].push_back(F(i,k));
                to_reserve++;
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

    void get_rotations(){
      Matrix3d S, U, V;
      int in, I, j;
      //current_energy = 0;
      RowVector3d v;
      for(int i = 0; i < Fixed.size(); i++){
        S.setZero();
        I = Fixed[i];
        for(int n = 0; n < N[I].size(); n++){
          j = N[I][n];
          v = V1.row(j);
          for(int k = 0; k < Fixed.size(); k++){
            if(j == Fixed[k]){
              v = V2.row(k);
            }
          }
          S += w_V1_diff[I].col(j) * (V2.row(i) - v); // V2_diff[i].col(j).transpose();

          //current_energy += W(i,j) * pow(( (v2i-v2j).transpose() - (R[i] * V1_diff[i].col(j))).norm(),2);
        }
        JacobiSVD<MatrixXd> svd(S, ComputeThinU | ComputeThinV);
        V = svd.matrixV(); U = svd.matrixU();
        R[I] = V*U.transpose();
        if(R[I].determinant() <= 0){
          svd.singularValues().minCoeff(&in);
          U.col(in) *= -1;
          R[I] = V*U.transpose();
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
      V1 = (chol.solve(b)).block(0, 0, n_vertices, 3);
    }


    MatrixXd ARAP(int n_iter, float crit){
      float old_energy;
      for(int iter = 0; iter < n_iter; iter++){
        old_energy = current_energy;
        get_rotations();
        get_p();
        if(abs(old_energy - current_energy) < crit){
          std::cout << "Iterations required: " << iter << std::endl;
          break;
        }
      }
      return V1;
    }

    MatrixXd ARAP(int n_iter){
      for(int iter = 0; iter < n_iter; iter++){
        get_rotations();
        get_p();
      }
      return V1;
    }

};