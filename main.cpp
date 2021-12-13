#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <vector>
#include <math.h>
#include <iostream>
#include<Eigen/Dense>
#include<fstream>
#include<Eigen/SparseCholesky>


#include <igl/min_quad_with_fixed.h>
#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject.h>
#include <igl/snap_points.h>
#include <igl/unproject_onto_mesh.h>
#include <Eigen/Core>
#include <stack>
#include <unistd.h>

using namespace Eigen;

typedef Triplet<double> T;

float calc_cotangent(Vector3d v1, Vector3d v2){
    return 1/tan(acos(v1.normalized().dot(v2.normalized())));
}

class Shape {
  public:
    MatrixXd V1, V2, W, b, b_top;
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


      b_top = MatrixXd::Zero(n_vertices, 3);
      int j;
      for(int i = 0; i < n_vertices; i++){
        for(int n = 0; n < N[i].size(); n++){
          j = N[i][n];
          b_top.row(i) += 0.5 * (R[i] + R[j]) * w_V1_diff[i].col(j);
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
      b.block(0,0,n_vertices,3) = b_top;
      for(int i = 0; i < n_fixed; i++){
        b.row(n_vertices + i) = V2.row(i);
      }
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
      int in, j;
      //current_energy = 0;
      RowVector3d v2i, v2j;
      for(int i = 0; i < n_vertices; i++){
        S.setZero();

        v2i = V1.row(i);
        for(int k; k < Fixed.size(); k++){
          if(i == Fixed[k]){
            v2i = V2.row(k);
          }
        }

        for(int n = 0; n < N[i].size(); n++){
          j = N[i][n];

          v2j = V1.row(j);
          for(int k; k < Fixed.size(); k++){
            if(i == Fixed[k]){
              v2j = V2.row(k);
            }
          }

          S += w_V1_diff[i].col(j) * (v2i - v2j); //V2.row(i) - V2.row(j) V2_diff[i].col(j).transpose();

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

// int main(int argc, char *argv[])
// {
//     MatrixXd Vo;
//     MatrixXi Fo;
//     // Load a mesh in OFF format
//     igl::readOFF("../meshes/bar1.off", Vo, Fo);
//     std::cout << Fo.rows() << std::endl;
//     // Plot the mesh

//     VectorXi Fixed(98);
//     for(int i = 0; i < 98; i++){
//       Fixed[i] = i;
//     }


//     MatrixXd Vd(98,3);
//     for(int i = 0; i < 49; i++){
//       Vd.row(i) = Vo.row(i);
//       Vd(i,0) += 100;
//       Vd(i,1) -= 50;
//     }
//     for(int i = 49; i < 98; i++){
//       Vd.row(i) = Vo.row(i);
//     }

//     Shape M(Vo,Fo);

//     M.fix(Fixed);

//     M.deform(Vd);

//     Vd = M.ARAP(10, 0.001);


//     //M.deform(Vd2, Fixed);
//     //Vd = M.ARAP(10);

//     igl::opengl::glfw::Viewer viewer;
//     //viewer.callback_key_down = &key_down;
//     viewer.data().set_mesh(Vd, Fo);
//     viewer.launch();
// }


// Undobale
struct State
{
  // Rest and transformed control points
  Eigen::MatrixXd CV, CU;
  bool placing_handles = true;
} s;

int main(int argc, char *argv[])
{
  // Undo Management
  std::stack<State> undo_stack,redo_stack;
  const auto push_undo = [&](State & _s=s)
  {
    undo_stack.push(_s);
    // clear
    redo_stack = std::stack<State>();
  };
  const auto undo = [&]()
  {
    if(!undo_stack.empty())
    {
      redo_stack.push(s);
      s = undo_stack.top();
      undo_stack.pop();
    }
  };
  const auto redo = [&]()
  {
    if(!redo_stack.empty())
    {
      undo_stack.push(s);
      s = redo_stack.top();
      redo_stack.pop();
    }
  };

  Eigen::MatrixXd V,U;
  Eigen::MatrixXi F;
  long sel = -1;
  Eigen::RowVector3f last_mouse;
  igl::min_quad_with_fixed_data<double> biharmonic_data, arap_data;
  Eigen::SparseMatrix<double> arap_K;

  // Load input meshes
  igl::read_triangle_mesh(
    (argc>1?argv[1]:"../meshes/bar1.off"),V,F);
  U = V;
  Shape M(V,F);
  igl::opengl::glfw::Viewer viewer;

  std::cout<<R"(
[click]  To place new control point
[drag]   To move control point
[space]  Toggle whether placing control points or deforming
U,u      Update deformation (i.e., run another iteration of solver)
R,r      Reset control points 
⌘ Z      Undo
⌘ ⇧ Z    Redo
)";


  const auto & update = [&]()
  {
    // predefined colors
    const Eigen::RowVector3d orange(1.0,0.7,0.2);
    const Eigen::RowVector3d yellow(1.0,0.9,0.2);
    const Eigen::RowVector3d blue(0.2,0.3,0.8);
    const Eigen::RowVector3d green(0.2,0.6,0.3);
    if(s.placing_handles)
    {
      viewer.data().set_vertices(V);
      viewer.data().set_colors(blue);
      viewer.data().set_points(s.CV,orange);
    }else
    {
      // SOLVE FOR DEFORMATION
      M.deform(s.CU);
      U = M.ARAP(1);
      //arap_single_iteration(arap_data,arap_K,s.CU,U);
      viewer.data().set_vertices(U);
      viewer.data().set_colors(orange);
      viewer.data().set_points(s.CU,blue);
    }
    viewer.data().compute_normals();
  };

  viewer.callback_mouse_down = 
    [&](igl::opengl::glfw::Viewer&, int, int)->bool
  {
    last_mouse = Eigen::RowVector3f(
      viewer.current_mouse_x,viewer.core().viewport(3)-viewer.current_mouse_y,0);
    if(s.placing_handles)
    {
      // Find closest point on mesh to mouse position
      int fid;
      Eigen::Vector3f bary;
      if(igl::unproject_onto_mesh(
        last_mouse.head(2),
        viewer.core().view,
        viewer.core().proj, 
        viewer.core().viewport, 
        V, F, 
        fid, bary))
      {
        long c;
        bary.maxCoeff(&c);
        Eigen::RowVector3d new_c = V.row(F(fid,c));
        if(s.CV.size()==0 || (s.CV.rowwise()-new_c).rowwise().norm().minCoeff() > 0)
        {
          push_undo();
          s.CV.conservativeResize(s.CV.rows()+1,3);
          // Snap to closest vertex on hit face
          s.CV.row(s.CV.rows()-1) = new_c;
          update();
          return true;
        }
      }
    }else
    {
      // Move closest control point
      Eigen::MatrixXf CP;
      igl::project(
        Eigen::MatrixXf(s.CU.cast<float>()),
        viewer.core().view,
        viewer.core().proj, viewer.core().viewport, CP);
      Eigen::VectorXf D = (CP.rowwise()-last_mouse).rowwise().norm();
      sel = (D.minCoeff(&sel) < 30)?sel:-1;
      if(sel != -1)
      {
        last_mouse(2) = CP(sel,2);
        push_undo();
        update();
        return true;
      }
    }
    return false;
  };

  viewer.callback_mouse_move = [&](igl::opengl::glfw::Viewer &, int,int)->bool
  {
    if(sel!=-1)
    {
      Eigen::RowVector3f drag_mouse(
        viewer.current_mouse_x,
        viewer.core().viewport(3) - viewer.current_mouse_y,
        last_mouse(2));
      Eigen::RowVector3f drag_scene,last_scene;
      igl::unproject(
        drag_mouse,
        viewer.core().view,
        viewer.core().proj,
        viewer.core().viewport,
        drag_scene);
      igl::unproject(
        last_mouse,
        viewer.core().view,
        viewer.core().proj,
        viewer.core().viewport,
        last_scene);
      s.CU.row(sel) += (drag_scene-last_scene).cast<double>();
      last_mouse = drag_mouse;
      update();
      return true;
    }
    return false;
  };
  viewer.callback_mouse_up = [&](igl::opengl::glfw::Viewer&, int, int)->bool
  {
    sel = -1;
    return false;
  };
  viewer.callback_key_pressed = 
    [&](igl::opengl::glfw::Viewer &, unsigned int key, int mod)
  {
    switch(key)
    {
      case 'R':
      case 'r':
      {
        push_undo();
        s.CU = s.CV;
        break;
      }
      case 'U':
      case 'u':
      {
        // Just trigger an update
        break;
      }
      case ' ':
        push_undo();
        s.placing_handles ^= 1;
        if(!s.placing_handles && s.CV.rows()>0)
        {
          // Switching to deformation mode
          s.CU = s.CV;
          Eigen::VectorXi b;
          igl::snap_points(s.CV,V,b);
          // PRECOMPUTATION FOR DEFORMATION
          M.fix(b);

          //arap_precompute(V,F,b,arap_data,arap_K);
        }
        break;
      default:
        return false;
    }
    update();
    return true;
  };

  // Special callback for handling undo
  viewer.callback_key_down = 
    [&](igl::opengl::glfw::Viewer &, unsigned char key, int mod)->bool
  {
    if(key == 'Z' && (mod & GLFW_MOD_SUPER))
    {
      (mod & GLFW_MOD_SHIFT) ? redo() : undo();
      update();
      return true;
    }
    return false;
  };
  viewer.callback_pre_draw = 
    [&](igl::opengl::glfw::Viewer &)->bool
  {
    if(viewer.core().is_animating && !s.placing_handles)
    {
      M.deform(s.CU);
      U = M.ARAP(1);
      //arap_single_iteration(arap_data,arap_K,s.CU,U);
      update();
    }
    return false;
  };
  viewer.data().set_mesh(V,F);
  viewer.data().show_lines = false;
  viewer.core().is_animating = true;
  viewer.data().face_based = true;
  update();
  viewer.launch();
  return EXIT_SUCCESS;
}