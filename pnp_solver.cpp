
#include "pnp_solver.h"
#include <eigen3/Eigen/Dense>

bool solvePnPbyDLT ( const Eigen::MatrixX3d& K, const std::vector< Eigen::Vector3d >& pts3d, const std::vector< Eigen::Vector2d >& pts2d, Eigen::Matrix3d& R, Eigen::Vector3d& t )
{
	// Check input
	if ( pts3d.size() != pts2d.size() || pts3d.size() < 6 ) {
		return false;
	}
	
	// Get camera params
	const double fx = K ( 0,0 );
	const double fy = K ( 1,1 );
	const double cx = K ( 0,2 );
	const double cy = K ( 1,2 );
	
	const int n = pts3d.size();
	
	/*Solve PnP by DLT */
	// Step 1. Construct matrix A, whose size is 2n x 12.
	Eigen::MatrixXd A;
	A.resize ( 2*n, 12 );
	for ( int i = 0; i < n; i ++ ) {
		const Eigen::Vector3d& pt3d = pts3d.at ( i );
		const Eigen::Vector2d& pt2d = pts2d.at ( i );
		
		const double& x = pt3d[0];
		const double& y = pt3d[1];
		const double& z = pt3d[2];
		const double& u = pt2d[0];
		const double& v= pt2d[1];
		
		A ( 2*i, 0 ) = x*fx;
		A ( 2*i, 1 ) = y*fx;
		A ( 2*i, 2 ) = z*fx;
		A ( 2*i, 3 ) = fx;
		A ( 2*i, 4 ) = 0.0;
		A ( 2*i, 5 ) = 0.0;
		A ( 2*i, 6 ) = 0.0;
		A ( 2*i, 7 ) = 0.0;
		A ( 2*i, 8 ) = x*cx-u*x;
		A ( 2*i, 9 ) = y*cx-u*y;
		A ( 2*i, 10 ) = z*cx-u*z;
		A ( 2*i, 11 ) = cx-u;
		
		
		A ( 2*i+1, 0 ) = 0.0;
		A ( 2*i+1, 1 ) = 0.0;
		A ( 2*i+1, 2 ) = 0.0;
		A ( 2*i+1, 3 ) = 0.0;
		A ( 2*i+1, 4 ) = x*fy;
		A ( 2*i+1, 5 ) = y*fy;
		A ( 2*i+1, 6 ) = z*fy;
		A ( 2*i+1, 7 ) = fy;
		A ( 2*i+1, 8 ) = x*cy-v*x;
		A ( 2*i+1, 9 ) = y*cy-v*y;
		A ( 2*i+1, 10 ) = z*cy-v*z;
		A ( 2*i+1, 11 ) = cy-v;
	} // construct matrix A.
	
	// Step 2. Solve Ax = 0 by SVD
	Eigen::JacobiSVD<Eigen::MatrixXd> svd_A ( A, Eigen::ComputeThinV );
	Eigen::MatrixXd V_A = svd_A.matrixV();
	
	// a1-a12 bar
	double a1 = V_A ( 0, 11 );
	double a2 = V_A ( 1, 11 );
	double a3 = V_A ( 2, 11 );
	double a4 = V_A ( 3, 11 );
	double a5 = V_A ( 4, 11 );
	double a6 = V_A ( 5, 11 );
	double a7 = V_A ( 6, 11 );
	double a8 = V_A ( 7, 11 );
	double a9 = V_A ( 8, 11 );
	double a10 = V_A ( 9, 11 );
	double a11 = V_A ( 10, 11 );
	double a12 = V_A ( 11, 11 );
	
	
	// Step 3. Reconstruct Rotation Matrix R and beta.
	Eigen::Matrix3d R_bar;
	R_bar << a1, a2, a3, a5, a6, a7, a9, a10, a11;
	Eigen::JacobiSVD<Eigen::MatrixXd> svd_R ( R_bar, Eigen::ComputeFullU | Eigen::ComputeFullV );
	Eigen::Matrix3d U_R = svd_R.matrixU();
	Eigen::Matrix3d V_R = svd_R.matrixV();
	Eigen::Vector3d V_Sigma = svd_R.singularValues();
	
	R = U_R * V_R.transpose();
	double beta = 1.0 / ( ( V_Sigma ( 0 ) +V_Sigma ( 1 ) +V_Sigma ( 2 ) ) / 3.0 );
	
	// Step 4. Compute t
	Eigen::Vector3d t_bar ( a4, a8, a12 );
	t = beta * t_bar;
	
	
	// Check + -
	int num_positive = 0;
	int num_negative = 0;
	for ( int i = 0; i < n; i ++ ) {
		const Eigen::Vector3d& pt3d = pts3d.at ( i );
		const double& x = pt3d[0];
		const double& y = pt3d[1];
		const double& z = pt3d[2];
		
		double lambda = beta * ( x * a9 + y* a10 + z* a11 + a12 );
		if ( lambda >= 0 ) {
			num_positive ++;
		} else {
			num_negative ++;
		}
	}
	
	if ( num_positive < num_negative ) {
		R = -R;
		t = -t;
	}
	
	return true;
}
