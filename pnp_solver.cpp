

#include "pnp_solver.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <eigen3/Eigen/QR>
#include <eigen3/Eigen/SVD>

bool solvePnPbyDLT ( const Eigen::Matrix3d& K, const std::vector< Eigen::Vector3d >& pts3d, const std::vector< Eigen::Vector2d >& pts2d, Eigen::Matrix3d& R, Eigen::Vector3d& t )
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


/******************************************
 * **************** EPnP ******************
 * ****************************************/

bool solvePnPbyEPnP ( const Eigen::Matrix3d& K, const std::vector< Eigen::Vector3d >& pts3d, const std::vector< Eigen::Vector2d >& pts2d, Eigen::Matrix3d& R, Eigen::Vector3d& t )
{
	// Check input data.
	if(pts2d.size() < 4 || pts2d.size() != pts3d.size())
		return false;
	
    // select control point in world coordinate frame.
    std::vector<Eigen::Vector3d> world_control_points;
    selectControlPoints ( pts3d, world_control_points );

    // compute homogeneous barycentric coordinates
    std::vector<Eigen::Vector4d> hb_coordinates;
    computeHomogeneousBarycentricCoordinates ( pts3d, world_control_points, hb_coordinates );

    // compute control point in camera coordinate frame.
    // construct Mx = 0
    Eigen::MatrixXd M;
    constructM ( K, hb_coordinates, pts2d, M );
    Eigen::Matrix<double, 12, 4> eigen_vectors;
    getFourEigenVectors ( M, eigen_vectors );

    // construct L * \beta = \rho.
    Eigen::Matrix<double, 6, 10> L;
    computeL ( eigen_vectors, L );

    Eigen::Matrix<double, 6, 1> rho;
    computeRho ( world_control_points, rho );

    // Case N = 2.
    Eigen::Vector4d betas;
    Eigen::Matrix3d tmp_R;
    Eigen::Vector3d tmp_t;

    {
		solveBetaN2 ( eigen_vectors, L, rho, betas );
        optimizeBeta ( L, rho, betas );

        std::vector<Eigen::Vector3d> camera_control_points;
        computeCameraControlPoints ( eigen_vectors, betas,  camera_control_points );

        std::vector<Eigen::Vector3d> pts3d_camera;
        rebuiltPts3dCamera ( camera_control_points, hb_coordinates,  pts3d_camera );

        computeRt ( pts3d_camera, pts3d, tmp_R, tmp_t );
    }
    double err2 = reprojectionError ( K, pts3d, pts2d, tmp_R, tmp_t );
    R = tmp_R;
    t = tmp_t;

    // Case N = 3.
    {
		solveBetaN3 (eigen_vectors, L, rho, betas );
        optimizeBeta ( L, rho, betas );

        std::vector<Eigen::Vector3d> camera_control_points;
        computeCameraControlPoints ( eigen_vectors, betas,  camera_control_points );

        std::vector<Eigen::Vector3d> pts3d_camera;
        rebuiltPts3dCamera ( camera_control_points, hb_coordinates,  pts3d_camera );

        computeRt ( pts3d_camera, pts3d, tmp_R, tmp_t );
    }
    double err3 = reprojectionError ( K, pts3d, pts2d, tmp_R, tmp_t );
    if ( err3 < err2 ) {
        R = tmp_R;
        t = tmp_t;
    }else{
		err3 = err2;
	}

    // Case N = 4.
    {
		solveBetaN4 ( eigen_vectors, L, rho, betas );
		optimizeBeta ( L, rho, betas );
		
		std::vector<Eigen::Vector3d> camera_control_points;
		computeCameraControlPoints ( eigen_vectors, betas,  camera_control_points );
		
		std::vector<Eigen::Vector3d> pts3d_camera;
		rebuiltPts3dCamera ( camera_control_points, hb_coordinates,  pts3d_camera );
		
		computeRt ( pts3d_camera, pts3d, tmp_R, tmp_t );
	}
	double err4 = reprojectionError ( K, pts3d, pts2d, tmp_R, tmp_t );
	if ( err4 < err3 ) {
		R = tmp_R;
		t = tmp_t;
	}

    return true;
} // solve PnP by EPnP

void selectControlPoints ( const std::vector< Eigen::Vector3d >& pts3d, std::vector< Eigen::Vector3d >& control_points )
{
    const int n = pts3d.size();
    control_points.reserve ( 4 );

    // select the center points
    Eigen::Vector3d cw1 ( 0.0, 0.0, 0.0 );
    for ( int i = 0; i < n; i ++ ) {
        cw1 += pts3d.at ( i );
    }
    cw1 /= ( double ) n;


    // PCA
    Eigen::MatrixXd A;
    A.resize ( n , 3 );
    for ( int i = 0; i < n; i ++ ) {
        A.block ( i, 0, 1, 3 ) = ( pts3d.at ( i ) - cw1 ).transpose();
    }
    Eigen::Matrix3d ATA = A.transpose() * A;

    Eigen::EigenSolver<Eigen::Matrix3d> es ( ATA );
    Eigen::Matrix3d D = es.pseudoEigenvalueMatrix();
    Eigen::Matrix3d V = es.pseudoEigenvectors();

    Eigen::Vector3d cw2 = cw1 + sqrt ( D ( 0,0 ) / n ) * V.block ( 0, 0, 3, 1 );
    Eigen::Vector3d cw3 = cw1 + sqrt ( D ( 1,1 ) / n ) * V.block ( 0, 1, 3, 1 );
    Eigen::Vector3d cw4 = cw1 + sqrt ( D ( 2,2 ) / n ) * V.block ( 0, 2, 3, 1 );

    control_points.push_back ( cw1 );
    control_points.push_back ( cw2 );
    control_points.push_back ( cw3 );
    control_points.push_back ( cw4 );
} // select control points


void computeHomogeneousBarycentricCoordinates ( const std::vector< Eigen::Vector3d >& pts3d, const std::vector< Eigen::Vector3d >& control_points, std::vector< Eigen::Vector4d >& hb_coordinates )
{
    const int n = pts3d.size();
    hb_coordinates.clear();
    hb_coordinates.reserve ( n );

    // construct C
    Eigen::Matrix4d C;
    for ( int i = 0; i < 4; i ++ ) {
        C.block ( 0, i, 3, 1 ) = control_points.at ( i );
    }
    C.block ( 3, 0, 1, 4 ) = Eigen::Vector4d ( 1.0, 1.0, 1.0, 1.0 );

    Eigen::Matrix4d C_inv = C.inverse();

    // compute \alpha_ij for all points
    for ( int i = 0; i < n; i ++ ) {
        Eigen::Vector4d ptw ( 0.0, 0.0, 0.0, 1.0 );
        ptw.block ( 0, 0, 3, 1 ) = pts3d.at ( i );
        hb_coordinates.push_back ( C_inv * ptw );
    }

} //computeHomogeneousBarycentricCoordinates


void constructM ( const Eigen::Matrix3d& K, const std::vector< Eigen::Vector4d >& hb_coordinates, const std::vector< Eigen::Vector2d >& pts2d, Eigen::MatrixXd& M )
{
    // get camera intrinsics
    const double fx = K ( 0,0 );
    const double fy = K ( 1,1 );
    const double cx = K ( 0,2 );
    const double cy = K ( 1,2 );

    // init M
    const int n = pts2d.size();
    M.resize ( 2*n, 12 );

    for ( int i = 0; i < n; i ++ ) {
        // get alphas
        const Eigen::Vector4d& alphas = hb_coordinates.at ( i );

        const double& alpha_i1 = alphas ( 0 );
        const double& alpha_i2 = alphas ( 1 );
        const double& alpha_i3 = alphas ( 2 );
        const double& alpha_i4 = alphas ( 3 );

        // get uv
        const double& u = pts2d.at ( i ) ( 0 );
        const double& v= pts2d.at ( i ) ( 1 );

        // idx
        const int id0 = 2*i;
        const int id1 = id0 +1;

        // the first line
        M ( id0, 0 ) = alpha_i1 * fx;
        M ( id0, 1 ) = 0.0;
        M ( id0, 2 ) = alpha_i1 * ( cx-u );

        M ( id0, 3 ) = alpha_i2 * fx;
        M ( id0, 4 ) = 0.0;
        M ( id0, 5 ) = alpha_i2 * ( cx-u );

        M ( id0, 6 ) = alpha_i3 * fx;
        M ( id0, 7 ) = 0.0;
        M ( id0, 8 ) = alpha_i3 * ( cx-u );

        M ( id0, 9 ) = alpha_i4 * fx;
        M ( id0, 10 ) = 0.0;
        M ( id0, 11 ) = alpha_i4 * ( cx-u );


        // for the second line
        M ( id1, 0 ) = 0.0;
        M ( id1, 1 ) = alpha_i1 * fy;
        M ( id1, 2 ) = alpha_i1 * ( cy-v );

        M ( id1, 3 ) = 0.0;
        M ( id1, 4 ) = alpha_i2 * fy;
        M ( id1, 5 ) = alpha_i2 * ( cy-v );

        M ( id1, 6 ) = 0.0;
        M ( id1, 7 ) = alpha_i3 * fy;
        M ( id1, 8 ) = alpha_i3 * ( cy-v );

        M ( id1, 9 ) = 0.0;
        M ( id1, 10 ) = alpha_i4 * fy;
        M ( id1, 11 ) = alpha_i4 * ( cy-v );
    } // Fill M.

} // construct M

void getFourEigenVectors ( const Eigen::MatrixXd& M, Eigen::Matrix< double, int ( 12 ), int ( 4 ) >& eigen_vectors )
{
    // Solve Mx = 0 by SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd ( M, Eigen::ComputeFullU | Eigen::ComputeFullV );
    Eigen::MatrixXd V = svd.matrixV();
    // Eigen::MatrixXd Sigma = svd.singularValues();
    // Eigen::Matrix3d U = svd.matrixU();
    // std::cout << Sigma.transpose() << std::endl;
    eigen_vectors = V.block ( 0, 8, 12, 4 );

// 	Eigen::Matrix<double, 12, 12> MTM = M.transpose() * M;
// 	Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 12, 12>> es(MTM);
//
// 	Eigen::VectorXd eigvec;
// 	Eigen::MatrixXd Vx = es.eigenvalues();
// 	Eigen::MatrixXd Dx = es.eigenvectors();
//
// 	std::cout << eigen_vectors << "\n\n";
// 	std::cout << Dx  << "\n\n\n\n";
}

void computeL ( const Eigen::Matrix< double, int ( 12 ), int ( 4 ) >& eigen_vectors, Eigen::Matrix< double, int ( 6 ), int ( 10 ) >& L )
{
    //[B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
    const int idx0[6] = {0, 0, 0, 1, 1, 2};
    const int idx1[6] = {1, 2, 3, 2, 3, 3};

    for ( int i = 0; i < 6; i ++ ) {
        const int idi = idx0[i] * 3;
        const int idj = idx1[i] * 3;

        // the first control point.
        const Eigen::Vector3d v1i = eigen_vectors.block ( idi, 3, 3, 1 );
        const Eigen::Vector3d v2i = eigen_vectors.block ( idi, 2, 3, 1 );
        const Eigen::Vector3d v3i = eigen_vectors.block ( idi, 1, 3, 1 );
        const Eigen::Vector3d v4i = eigen_vectors.block ( idi, 0, 3, 1 );

        // the second control point
        const Eigen::Vector3d v1j = eigen_vectors.block ( idj, 3, 3, 1 );
        const Eigen::Vector3d v2j = eigen_vectors.block ( idj, 2, 3, 1 );
        const Eigen::Vector3d v3j = eigen_vectors.block ( idj, 1, 3, 1 );
        const Eigen::Vector3d v4j = eigen_vectors.block ( idj, 0, 3, 1 );

        Eigen::Vector3d S1 = v1i - v1j;
        Eigen::Vector3d S2 = v2i - v2j;
        Eigen::Vector3d S3 = v3i - v3j;
        Eigen::Vector3d S4 = v4i - v4j;

        Eigen::Matrix<double, 1, 3> S1_T = S1.transpose();
        Eigen::Matrix<double, 1, 3> S2_T = S2.transpose();
        Eigen::Matrix<double, 1, 3> S3_T = S3.transpose();
        Eigen::Matrix<double, 1, 3> S4_T = S4.transpose();

        //[B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
        L ( i, 0 ) = S1_T * S1;
        L ( i, 1 ) = 2 * S1_T * S2;
        L ( i, 2 ) = S2_T * S2;
        L ( i, 3 ) = 2 * S1_T * S3;
        L ( i, 4 ) = 2 * S2_T * S3;
        L ( i, 5 ) = S3_T * S3;
        L ( i, 6 ) = 2 * S1_T * S4;
        L ( i, 7 ) = 2 * S2_T * S4;
        L ( i, 8 ) = 2 * S3_T * S4;
        L ( i, 9 ) = S4_T * S4;
    }
} // compute L


void computeRho ( const std::vector< Eigen::Vector3d >& control_points, Eigen::Matrix<double, 6, 1>& rho )
{
    const int idx0[6] = {0, 0, 0, 1, 1, 2};
    const int idx1[6] = {1, 2, 3, 2, 3, 3};
    for ( int i = 0; i < 6; i ++ ) {
        Eigen::Vector3d v01 = control_points.at ( idx0[i] ) - control_points.at ( idx1[i] );
        rho ( i, 0 ) = ( v01.transpose() * v01 );
    }
}

void solveBetaN2 (const Eigen::Matrix<double, 12, 4>& eigen_vectors,  const Eigen::Matrix< double, int ( 6 ), int ( 10 ) >& L, const Eigen::Matrix<double, 6, 1>& rho, Eigen::Vector4d& betas )
{
    //                  [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
    // betas_approx_2 = [B11 B12 B22                            ]

    const Eigen::Matrix<double, 6, 3>& L_approx = L.block ( 0, 0, 6, 3 );
    Eigen::Vector3d b3 = L_approx.fullPivHouseholderQr().solve ( rho );

    if ( b3[0] < 0 ) {
        betas[0] = sqrt ( -b3[0] );
        betas[1] = ( b3[2] < 0 ) ? sqrt ( -b3[2] ) : 0.0;
    } else {
        betas[0] = sqrt ( b3[0] );
        betas[1] = ( b3[2] > 0 ) ? sqrt ( b3[2] ) : 0.0;
    }

    if ( b3[1] < 0 ) {
        betas[0] = -betas[0];
    }

    betas[2] = 0.0;
    betas[3] = 0.0;
	
	// Check betas.
	std::vector<Eigen::Vector3d> camera_control_points;
	computeCameraControlPoints(eigen_vectors, betas, camera_control_points);
	if(isGoodBetas(camera_control_points) == false)
	{
		betas = -betas;
	}
}

void solveBetaN3 (const Eigen::Matrix<double, 12, 4>& eigen_vectors,  const Eigen::Matrix< double, int ( 6 ), int ( 10 ) >& L, const Eigen::Matrix< double, int ( 6 ), int ( 1 ) >& rho, Eigen::Vector4d& betas )
{
    //                  [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
    // betas_approx_3 = [B11 B12 B22 B13 B23                    ]

    const Eigen::Matrix<double, 6, 5>& L_approx = L.block ( 0, 0, 6, 5 );
    Eigen::Matrix<double, 5, 1> b5 = L_approx.fullPivHouseholderQr().solve ( rho );

    if ( b5[0] < 0 ) {
        betas[0] = sqrt ( -b5[0] );
        betas[1] = ( b5[2] < 0 ) ? sqrt ( -b5[2] ) : 0.0;
    } else {
        betas[0] = sqrt ( b5[0] );
        betas[1] = ( b5[2] > 0 ) ? sqrt ( b5[2] ) : 0.0;
    }
    if ( b5[1] < 0 ) {
        betas[0] = -betas[0];
    }
    betas[2] = b5[3] / betas[0];
    betas[3] = 0.0;
	
	// Check betas.
	std::vector<Eigen::Vector3d> camera_control_points;
	computeCameraControlPoints(eigen_vectors, betas, camera_control_points);
	if(isGoodBetas(camera_control_points) == false)
	{
		betas = -betas;
	}
}

void solveBetaN4 (const Eigen::Matrix<double, 12, 4>& eigen_vectors,  const Eigen::Matrix< double, int ( 6 ), int ( 10 ) >& L, const Eigen::Matrix< double, int ( 6 ), int ( 1 ) >& rho, Eigen::Vector4d& betas )
{
    // betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
    // betas_approx_1 = [B11 B12     B13         B14]

    Eigen::Matrix<double, 6, 4> L_approx;
    L_approx.block ( 0,0, 6, 2 ) = L.block ( 0,0, 6,2 );
    L_approx.block ( 0,2, 6, 1 ) = L.block ( 0,3, 6,1 );
    L_approx.block ( 0,3, 6, 1 ) = L.block ( 0,6, 6,1 );

    Eigen::Vector4d b4 = L_approx.fullPivHouseholderQr().solve ( rho );

    if ( b4[0] < 0 ) {
        betas[0] = sqrt ( -b4[0] );
        betas[1] = -b4[1] / betas[0];
        betas[2] = -b4[2] / betas[0];
        betas[3] = -b4[3] / betas[0];
    } else {
        betas[0] = sqrt ( b4[0] );
        betas[1] = b4[1] / betas[0];
        betas[2] = b4[2] / betas[0];
        betas[3] = b4[3] / betas[0];
    }
    
    // Check betas.
    std::vector<Eigen::Vector3d> camera_control_points;
	computeCameraControlPoints(eigen_vectors, betas, camera_control_points);
	if(isGoodBetas(camera_control_points) == false)
	{
		betas = -betas;
	}
    
}

void optimizeBeta ( const Eigen::Matrix< double, int ( 6 ), int ( 10 ) >& L, const Eigen::Matrix< double, int ( 6 ), int ( 1 ) >& rho, Eigen::Vector4d& betas )
{
    const int iter_num = 5;

    for ( int nit = 0; nit < iter_num; nit ++ ) {

        // construct J
        Eigen::Matrix<double, 6, 4> J;
        for ( int i = 0; i < 6; i ++ ) {
            // [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
            J ( i, 0 ) = 2 * betas[0] * L ( i, 0 ) + betas[1]* L ( i, 1 ) + betas[2]*L ( i, 3 ) + betas[3]*L ( i, 6 );
            J ( i, 1 ) = betas[0] * L ( i, 1 ) + 2 * betas[1]* L ( i, 2 ) + betas[2]*L ( i, 3 ) + betas[3]*L ( i, 7 );
            J ( i, 2 ) = betas[0] * L ( i, 3 ) + betas[1]* L ( i, 4 ) + 2 * betas[2]*L ( i, 5 ) + betas[3]*L ( i, 8 );
            J ( i, 3 ) = betas[0] * L ( i, 6 ) + betas[1]* L ( i, 7 ) + betas[2]*L ( i, 8 ) + 2 * betas[3]*L ( i, 9 );
        }

        Eigen::Matrix<double, 4, 6> J_T = J.transpose();
        Eigen::Matrix<double, 4, 4> H = J_T * J;

        // Compute residual
        // [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
        // [B00 B01 B11 B02 B12 B22 B03 B13 B23 B33]
        Eigen::Matrix<double, 10, 1> bs;
        bs << betas[0]*betas[0], betas[0]*betas[1], betas[1]*betas[1], betas[0]*betas[2], betas[1]*betas[2],
           betas[2]*betas[2], betas[0]*betas[3], betas[1]*betas[3], betas[2]*betas[3], betas[3]*betas[3];
        Eigen::Matrix<double, 6, 1> residual = L * bs - rho;

        //std::cout << "Error " << residual.transpose() * residual << "\n";

        // Solve J^T * J \delta_beta = -J^T * residual;
        Eigen::Matrix<double, 4, 1> delta_betas = H.fullPivHouseholderQr().solve ( -J_T * residual );

        // update betas;
        betas += delta_betas;
    } //iter n times.
} //


void computeCameraControlPoints ( const Eigen::Matrix< double, int ( 12 ), int ( 4 ) >& eigen_vectors, const Eigen::Vector4d& betas, std::vector< Eigen::Vector3d >& camera_control_points )
{
    camera_control_points.clear();
    camera_control_points.reserve ( 4 );

    Eigen::Matrix<double, 12, 1> vec =
        betas[0] * eigen_vectors.block ( 0, 3, 12, 1 ) +
        betas[1] * eigen_vectors.block ( 0, 2, 12, 1 ) +
        betas[2] * eigen_vectors.block ( 0, 1, 12, 1 ) +
        betas[3] * eigen_vectors.block ( 0, 0, 12, 1 );

    for ( int i = 0; i < 4; i ++ ) {
        camera_control_points.push_back ( vec.block ( i*3 ,0, 3, 1 ) );
    }
}


bool isGoodBetas(const std::vector<Eigen::Vector3d>& camera_control_points)
{
	int num_positive = 0;
	int num_negative = 0;
	
	for(int i  =0; i < 4; i ++)
	{
		if(camera_control_points.at(i)[2] > 0)
		{
			num_positive++;
		}
		else
		{
			num_negative ++;
		}
	}
	
	if(num_negative >= num_positive)
	{
		return false;
	}
	
	return true;
}


void rebuiltPts3dCamera ( const std::vector< Eigen::Vector3d >& camera_control_points, const std::vector< Eigen::Vector4d >& hb_coordinates, std::vector< Eigen::Vector3d >& pts3d_camera )
{
    const int n = hb_coordinates.size();
    pts3d_camera.clear();
    pts3d_camera.reserve ( n );

    for ( int i = 0; i < n; i ++ ) {
        Eigen::Vector4d alphas = hb_coordinates.at ( i );

        Eigen::Vector3d ptc =
            camera_control_points[0] * alphas[0]+
            camera_control_points[1] * alphas[1]+
            camera_control_points[2] * alphas[2]+
            camera_control_points[3] * alphas[3];
        pts3d_camera.push_back ( ptc );
    }
}

void computeRt ( const std::vector< Eigen::Vector3d >& pts3d_camera, const std::vector< Eigen::Vector3d >& pts3d_world, Eigen::Matrix3d& R, Eigen::Vector3d& t )
{
    const int n = pts3d_camera.size();
    // step 1. compute center points
    Eigen::Vector3d pcc ( 0.0,0.0,0.0 );
    Eigen::Vector3d pcw ( 0.0,0.0,0.0 );

    for ( int i = 0; i < n; i++ ) {
        pcc += pts3d_camera.at ( i );
        pcw += pts3d_world.at ( i );
    }

    pcc /= ( double ) n;
    pcw /= ( double ) n;

    // step 2. remove centers.
    Eigen::MatrixXd Pc, Pw;
    Pc.resize ( n, 3 );
    Pw.resize ( n, 3 );

    for ( int i = 0; i < n; i ++ ) {
        Pc.block ( i, 0, 1, 3 ) = ( pts3d_camera.at ( i ) - pcc ).transpose();

        Pw.block ( i, 0, 1, 3 ) = ( pts3d_world.at ( i ) - pcw ).transpose();
    }

    // step 3. compute R.
    Eigen::Matrix3d W = Pc.transpose() * Pw;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd ( W, Eigen::ComputeFullU | Eigen::ComputeFullV );
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    R = U * V.transpose();

    if ( R.determinant() < 0 ) {
        R.block ( 2, 0, 1, 3 ) = - R.block ( 2, 0, 1, 3 );
    }

    // step 3. compute t
    t = pcc - R * pcw;
}


double reprojectionError ( const Eigen::Matrix3d& K, const std::vector< Eigen::Vector3d >& pts3d_world, const std::vector<Eigen::Vector2d>& pts2d, const Eigen::Matrix3d& R, const Eigen::Vector3d& t )
{
    const int n = pts3d_world.size();
    double sum_err2 = 0.0;
    for ( size_t i = 0; i < pts3d_world.size(); i ++ ) {
        const Eigen::Vector3d& ptw = pts3d_world.at ( i );
        Eigen::Vector3d lamda_uv = K * ( R * ptw + t );
        Eigen::Vector2d uv = lamda_uv.block ( 0,0, 2, 1 ) / lamda_uv ( 2 );

        Eigen::Vector2d e_uv = pts2d.at ( i ) - uv;
        sum_err2 += e_uv.transpose() * e_uv;
    }

    return sqrt ( sum_err2 / ( double ) n );
}




