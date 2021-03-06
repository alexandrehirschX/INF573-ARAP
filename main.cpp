
#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/snap_points.h>

#include "MeshTransformation.cpp"
#include "shape.cpp"

using namespace Eigen;

RowVector3d compute_barycenter(MatrixXd &V)
{
  RowVector3d bar(0., 0., 0.);
  int n = V.rows();
  for(int i = 0; i < n;i++){
    bar += V.row(i);
  }
  bar /= n;
  return bar;
}


struct State
{
  // Rest and transformed control points
  // Only the orange vertices of CV1 will be able to move, the others are fixed, CU are the transformed CV1 points
  // CC is the colour matrix, and CI holds the indices of points in CC that are coloured orange, i.e (can move)
  MatrixXd CV1, CC, CI, CU;
  // When true we can place orange control points 
  bool placing_handles = true;
  // When trye we can place red control points
  bool placing_cp = false;
} s;



int main(int argc, char *argv[]){

  int dir = 0;
  MeshTransformation rotationXp(0.2, 0);
  MeshTransformation rotationYp(0.2, 1);
  MeshTransformation rotationZp(0.2, 2);
  std::vector<MeshTransformation> rotationp{rotationXp, rotationYp, rotationZp};

  MeshTransformation rotationXm(-0.2, 0);
  MeshTransformation rotationYm(-0.2, 1);
  MeshTransformation rotationZm(-0.2, 2);
  std::vector<MeshTransformation> rotationm{rotationXm, rotationYm, rotationZm};


  MatrixXd V,U;
  MatrixXi F;
  
  long sel = -1;
  RowVector3f last_mouse;

  // Load input meshes
  igl::read_triangle_mesh((argc>1?argv[1]:"../meshes/bar3.off"),V,F);
  bool b = argc>2? (bool) argv[2]: false;
  std::cout << V.cols() << std::endl;
  U = V;
  std::cout << "Loading Shape ... " << std::endl;
  Shape M(V,F);
  std::cout << "Done" << std::endl;
  igl::opengl::glfw::Viewer viewer;

  std::cout << R"(
  [click]  To place new control point
  [drag]   To move handle points
  [space]  Toggle whether placing points or deforming
  C,c      Toggle whether placing fixed or handle control points
  V,v      Toggle rotation direction (X, Y, Z)
  B,b      Positive angle rotation
  N,n      Negative angle rotation 
  U,u      Update deformation (i.e., run another iteration of solver)
  R,r      Reset control points 
  )";


  const auto & update = [&]()
  {
    // predefined colors
    const RowVector3d orange(1.0,0.7,0.2);
    const RowVector3d red(1.0,0.0,0.0);
    const RowVector3d yellow(1.0,0.9,0.2);
    const RowVector3d blue(0.2,0.3,0.8);
    const RowVector3d green(0.2,0.6,0.3);

    if(s.placing_handles)
    {
      viewer.data().set_vertices(V);
      viewer.data().set_colors(blue);
      viewer.data().set_points(s.CV1, s.CC);
    }
    else if(s.placing_cp)
    {
      viewer.data().set_vertices(V);
      viewer.data().set_colors(green);
      viewer.data().set_points(s.CV1, s.CC);
    }
    else{
      // SOLVE FOR DEFORMATION
      M.deform(s.CU);
      M.ARAP(U,1);
      viewer.data().set_vertices(U);
      viewer.data().set_colors(orange);
      viewer.data().set_points(s.CU, s.CC);
    }
    viewer.data().compute_normals();
  };

  viewer.callback_mouse_down = 
    [&](igl::opengl::glfw::Viewer&, int, int)->bool
  {
    const RowVector3d orange(1.0,0.7,0.2);
    const RowVector3d red(1.0,0.0,0.0);
    last_mouse = RowVector3f(
      viewer.current_mouse_x,viewer.core().viewport(3)-viewer.current_mouse_y,0);
    if(s.placing_handles){
      // Find closest point on mesh to mouse position
      int fid;
      Vector3f bary;
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
        RowVector3d new_c = V.row(F(fid,c));
        if(s.CV1.size()==0 || (s.CV1.rowwise()-new_c).rowwise().norm().minCoeff() > 0)
        {
          // Snap to closest vertex on hit face
          s.CV1.conservativeResize(s.CV1.rows()+1,3);
          s.CV1.row(s.CV1.rows()-1) = new_c;

          s.CC.conservativeResize(s.CC.rows()+1,3);
          s.CC.row(s.CC.rows() -1) = orange;

          // Keeping track of this orange point
          s.CI.conservativeResize(s.CI.rows()+1, 1);
          s.CI(s.CI.rows()-1, 0) = s.CV1.rows()-1;

          update();
          return true;
        }
      }
    }
    // Setting control points that will NOT move 
    else if(s.placing_cp) {
      // Find closest point on mesh to mouse position
      int fid;
      Vector3f bary;
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
        RowVector3d new_c = V.row(F(fid,c));
        if(s.CV1.size()==0 || (s.CV1.rowwise()-new_c).rowwise().norm().minCoeff() > 0)
        {
          // Snap to closest vertex on hit face
          s.CV1.conservativeResize(s.CV1.rows()+1,3);
          s.CV1.row(s.CV1.rows()-1) = new_c;

          s.CC.conservativeResize(s.CC.rows()+1,3);
          s.CC.row(s.CC.rows() -1) = red;

          update();
          return true;
        }
      }
    }
    else{
      // Move closest control point
      MatrixXf CP;
      igl::project(
        MatrixXf(s.CU.cast<float>()),
        viewer.core().view,
        viewer.core().proj, viewer.core().viewport, CP);
      VectorXf D = (CP.rowwise()-last_mouse).rowwise().norm();
      sel = (D.minCoeff(&sel) < 30)?sel:-1;
      if(sel != -1)
      {
        last_mouse(2) = CP(sel,2);
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
      RowVector3f drag_mouse(
        viewer.current_mouse_x,
        viewer.core().viewport(3) - viewer.current_mouse_y,
        last_mouse(2));
      RowVector3f drag_scene,last_scene;
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

      for (int i = 0; i < s.CI.rows(); i  ++){
        // Index of orange point in s.CV1
        int idx = s.CI(i, 0);
        s.CU.row(idx) += (drag_scene-last_scene).cast<double>();
      }
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
        s.CU = s.CV1;
        break;
      }
      case 'U':
      case 'u':
      {
        // Just trigger an update
        break;
      }
      case 'C' :
      case 'c' :
      {
        s.placing_cp ^=1;
        s.placing_handles ^= 1;
        break;
      }
      case ' ':{
        s.placing_handles ^= 1;
        if(!s.placing_handles && s.CV1.rows()>0)
        {
          // Switching to deformation mode
          s.CU = s.CV1;

          VectorXi b;
          igl::snap_points(s.CV1,V,b);
          // PRECOMPUTATION FOR DEFORMATION
          M.fix(b);
        }
        break;
      }

      case 'V':
      case 'v':{
        // Changing direction of 
        dir += 1;
        if(dir == 3) dir = 0; 
        break;
      }
      case 'B':
      case 'b':{
        MatrixXd bb(s.CI.rows(), 3);
        for(int i = 0; i < bb.rows(); i++){
          int idx = s.CI(i, 0);
          bb.row(i) = s.CU.row(idx);
        }
        rotationp[dir].barycenter = compute_barycenter(bb);
        rotationp[dir].transform(bb);
        for(int i = 0; i < bb.rows(); i++){
          int idx = s.CI(i, 0);
          s.CU.row(idx) = bb.row(i);
        }
        break;
      }
      case 'N':
      case 'n':{
        MatrixXd nn(s.CI.rows(), 3);
        for(int i = 0; i < nn.rows(); i++)
        {
          int idx = s.CI(i, 0);
          nn.row(i) = s.CU.row(idx);
        }
        rotationm[dir].barycenter = compute_barycenter(nn);
        rotationm[dir].transform(nn);
        for(int i = 0; i < nn.rows(); i++){
          int idx = s.CI(i, 0);
          s.CU.row(idx) = nn.row(i);
        }
        break;
      }
      default:
        return false;
    }
    update();
    return true;
  };

  viewer.callback_pre_draw = 
    [&](igl::opengl::glfw::Viewer &)->bool
  {
    if(viewer.core().is_animating && !s.placing_handles && !s.placing_cp)
    {
      M.deform(s.CU);
      if(M.ARAP(U,1,1)) update();
    }
    return false;
  };
  viewer.data().set_mesh(V,F);
  viewer.data().show_lines = true;
  viewer.core().is_animating = b; //true;
  viewer.data().face_based = true;
  update();
  
  viewer.launch();
  return EXIT_SUCCESS;
}