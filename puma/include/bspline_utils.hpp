/* ----------------------------------------------------------------------------
 * Copyright 2021, Jesus Tordesillas Torres, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Jesus Tordesillas, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/Splines>
#include "panther_types.hpp"

Eigen::RowVectorXd constructKnotsClampedUniformSpline(double t_init, double t_end, int deg, int num_seg);

void CPs2Traj(std::vector<Eigen::Vector3d> &qp, std::vector<double> &qy, Eigen::RowVectorXd &knots_p,
              Eigen::RowVectorXd &knots_y, std::vector<mt::state> &traj, int deg_p, int deg_y, double dc);
void CPs2Traj(std::vector<Eigen::Vector2d> &qp, std::vector<double> &qy, Eigen::RowVectorXd &knots_p,
              Eigen::RowVectorXd &knots_y, std::vector<mt::state> &traj, int deg_p, int deg_y, double dc);

mt::state getStatePosSplineT(const std::vector<Eigen::Vector2d> &qp, const Eigen::RowVectorXd &knots_p, int deg_p,
                             double t);

Eigen::Spline3d findInterpolatingBsplineNormalized(const std::vector<double> &times,
                                                   const std::vector<Eigen::Vector3d> &positions);