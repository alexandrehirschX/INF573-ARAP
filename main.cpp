
#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/snap_points.h>
#include <stack>

#include "shape.cpp"

using namespace Eigen;


struct State
{
  // Rest and transformed control points
  // CV1 will move, CV2 will be non-moving control points, CU are the transformed CV1 points
  MatrixXd CV1, CV2, CU;
  bool placing_handles = true;
  // boolean for if you're placing control points that you absolutely do not want to move
  bool placing_cp = false;
} s;



int main(int argc, char *argv[]){

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

  MatrixXd V,U;
  MatrixXi F;
  long sel = -1;
  RowVector3f last_mouse;

  // Load input meshes
  igl::read_triangle_mesh((argc>1?argv[1]:"../meshes/bar3.off"),V,F);
  U = V;
  std::cout << "Loading Shape ... " << std::endl;
  Shape M(V,F);
  std::cout << "Done" << std::endl;
  igl::opengl::glfw::Viewer viewer;

  std::cout << R"(
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
    const RowVector3d orange(1.0,0.7,0.2);
    const RowVector3d red(1.0,0.0,0.0);
    const RowVector3d yellow(1.0,0.9,0.2);
    const RowVector3d blue(0.2,0.3,0.8);
    const RowVector3d green(0.2,0.6,0.3);
    if(s.placing_handles)
    {
      viewer.data().set_vertices(V);
      viewer.data().set_colors(blue);
      viewer.data().set_points(s.CV1, orange);
      if (s.CV2.size() != 0){
        viewer.data().set_points(s.CV2, red);
      }
    }
    else if (s.placing_cp) {
      viewer.data().set_vertices(V);
      viewer.data().set_colors(blue);
      std::cout << "Hello" << std::endl;
      viewer.data().set_points(s.CV1, orange);
      viewer.data().set_points(s.CV2, red);
      // if (s.CV1.size() != 0){
      //   std::cout << "i made it here" << std::endl;
      //   viewer.data().set_points(s.CV1, orange);
      // }
    }
    else{
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
          push_undo();
          s.CV1.conservativeResize(s.CV1.rows()+1,3);
          // Snap to closest vertex on hit face
          s.CV1.row(s.CV1.rows()-1) = new_c;
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
        if(s.CV2.size()==0 || (s.CV2.rowwise()-new_c).rowwise().norm().minCoeff() > 0)
        {
          push_undo();
          s.CV2.conservativeResize(s.CV2.rows()+1,3);
          // Snap to closest vertex on hit face
          s.CV2.row(s.CV2.rows()-1) = new_c;
          std::cout << s.CV1.rows() << std::endl;
          std::cout << s.placing_handles << std::endl;
          std::cout << s.placing_cp << std::endl;d:
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
      // s.CU.row(sel) += (drag_scene-last_scene).cast<double>();
      /// HARD CODED HERE
      for (int i = 0; i < 12; i ++ ){
        s.CU.row(i) += (drag_scene-last_scene).cast<double>();
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
        push_undo();
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
        push_undo();
        s.placing_cp ^=1;
        s.placing_handles ^= 1;
        break;
      }
      case ' ':
        push_undo();
        s.placing_handles ^= 1;
        if(!s.placing_handles && s.CV1.rows()>0)
        {
          // Switching to deformation mode
          s.CU = s.CV1;
          VectorXi b;
          igl::snap_points(s.CV1,V,b);
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
  viewer.core().is_animating = false; //true;
  viewer.data().face_based = true;
  update();
  viewer.launch();
  return EXIT_SUCCESS;
}