#include <igl/opengl/glfw/Viewer.h>
#include <ostream>
#include <cmath>

using namespace Eigen;

/**
 * A class for representing linear transformations on 3D points (using homogeneous coordinates)
 * */
class MeshTransformation
{
public:
RowVector3d barycenter;
/*
Initialize the identity transformation
**/
  MeshTransformation()
  {
    MatrixXd m(4, 4);
    m(0, 0) = 1.0; m(1, 0) = 0.0; m(2, 0) = 0.0; m(3, 0) = 0.0;
    m(0, 1) = 0.0; m(1, 1) = 1.0; m(2, 1) = 0.0; m(3, 1) = 0.0;
    m(0, 2) = 0.0; m(1, 2) = 0.0; m(2, 2) = 1.0; m(3, 2) = 0.0;
    m(0, 3) = 0.0; m(1, 3) = 0.0; m(2, 3) = 0.0; m(3, 3) = 1.0;

    M = m;
  }


/*
Initialize a scaling transformation
**/
  MeshTransformation(double s1, double s2, double s3)
  {
    // TO BE COMPLETED
    set_matrix(MeshTransformation().M);
    M(0,0) = s1;
    M(1,1) = s2;
    M(2,2) = s3;
  }

/*
Initialize a rotation transformation around a given axis (X, Y or Z) <br><br>

 @param  direction  a value 0, 1 or 2 indicating the direction (X, Y or Z respectively)
**/
  MeshTransformation(double theta, int direction)
  {
    // TO BE COMPLETED

    set_matrix(MeshTransformation().M);
    
    double c = cos(theta);
    double s = sin(theta);

    if(direction == 0){
      M(1, 1) = c;
      M(2, 1) = s;
      M(1, 2) = -s;
      M(2, 2) = c;
    }
    else if(direction == 1){
      M(0, 0) = c;
      M(2, 0) = -s;
      M(0, 2) = s;
      M(2, 2) = c;
    }
    else if(direction == 2){ 
      M(0, 0) = c;
      M(1, 0) = s;
      M(0, 1) = -s;
      M(1, 1) = c;
    }

  }

/*
Initialize a translation
**/
  MeshTransformation(RowVector3d t)
  {
    // TO BE COMPLETED
    set_matrix(MeshTransformation().M);
    M(3,0) = t.x();
    M(3,1) = t.y();
    M(3,2) = t.z();
  }

/*
Matrix accessor

@return  the matrix transformation
**/
  MatrixXd get_matrix() {
    return M;
  }

/*
Initialize a transformation given an input matrix 'm'
**/
  void set_matrix(MatrixXd m)
  {
    M = m;
  }

/*
Apply the transformation to all vertices stored in a matrix 'V' <br>

@param V  vector storing the input points
**/

  void transform(MatrixXd &V) {
    
    // TO BE COMPLETED

    for(int i = 0; i < V.rows(); i++){
      V.row(i) = transform(V.row(i));
    }
  }

  	/**
	 * Apply the transformation to a 3d (row) vector 'v' <br>
   * 
   * Remark: use homogeneous coordinates
   * 
   * @return  the vector after transformation
	 */
  
	RowVector3d transform(RowVector3d v) {
    // TO BE COMPLETED
    
    v -= barycenter;
    RowVector4d s(v.x(),v.y(),v.z(),1.);
    s = s*M;
    RowVector3d result(s.x(), s.y(), s.z());
    result += barycenter;
    return result;
	}

	/**
	 * Compose the current transformation with a transfomation 't': return a new transformation
	 */
  
	MeshTransformation compose(MeshTransformation t) {
    
    MeshTransformation res(1.0, 1.0, 1.0);
    res.set_matrix(M*t.M);
    // TO BE COMPLETED
    return res;
    
	}

	/**
	 * Print the matrix transformation
	 */
  friend std::ostream& operator<<(std::ostream &os, MeshTransformation& t) {
    return os << "matrix:\n" << t.get_matrix() << std::endl;
  }

private:
  MatrixXd M; // a 4x4 matrix representing a linear transformation
};

