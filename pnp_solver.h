
#ifndef PNP_SOLVER_H
#define PNP_SOLVER_H

#include <eigen3/Eigen/Core>
#include <vector>

bool solvePnPbyDLT( const Eigen::MatrixX3d& K, const std::vector<Eigen::Vector3d>& pts3d, const std::vector<Eigen::Vector2d>& pts2d, Eigen::Matrix3d& R, Eigen::Vector3d& t );


#endif // PNP_SOLVER_H