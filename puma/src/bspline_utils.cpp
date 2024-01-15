/* ----------------------------------------------------------------------------
 * Copyright 2021, Jesus Tordesillas Torres, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Jesus Tordesillas, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

#include "bspline_utils.hpp"
#include <cassert>

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

Eigen::RowVectorXd constructKnotsClampedUniformSpline(double t_init, double t_end, int deg, int num_seg)
{
  int p = deg;
  int M = num_seg + 2 * p;
  // int num_seg = M - 2 * p;
  double deltaT = (t_end - t_init) / num_seg;

  // std::cout << "num_seg= " << num_seg << std::endl;
  // std::cout << "deltaT= " << deltaT << std::endl;
  // std::cout << "p= " << p << std::endl;
  // std::cout << "M= " << M << std::endl;

  Eigen::RowVectorXd knots(M + 1);

  for (int i = 0; i <= p; i++)
  {
    knots[i] = t_init;
  }

  for (int i = (p + 1); i <= M - p - 1; i++)
  {
    knots[i] = knots[i - 1] + deltaT;  // Uniform b-spline (internal knots are equally spaced)
  }

  for (int i = (M - p); i <= M; i++)
  {
    knots[i] = t_end;
  }

  return knots;
}

//
mt::state getStatePosSplineT(const std::vector<Eigen::Vector2d> &qp, const Eigen::RowVectorXd &knots_p, int deg_p,
                             double t)
{
  assert(((knots_p.size() - 1) == (qp.size() - 1) + deg_p + 1) && "M=N+p+1 not satisfied");

  int num_seg = (knots_p.size() - 1) - 2 * deg_p;  // M-2*p

  // Stack the control points in matrices
  Eigen::Matrix<double, 2, -1> qp_matrix(2, qp.size());
  for (int i = 0; i < qp.size(); i++)
  {
    qp_matrix.col(i) = qp[i];
  }

  // Construct now the B-Spline, see https://github.com/libigl/eigen/blob/master/unsupported/test/splines.cpp#L37
  Eigen::Spline<double, 2, Eigen::Dynamic> spline_p(knots_p, qp_matrix);

  Eigen::MatrixXd derivatives_p = spline_p.derivatives(t, 4);  // compute the derivatives up to that order

  mt::state state_i;

  state_i.setPos(derivatives_p.col(0));  // First column
  state_i.setVel(derivatives_p.col(1));
  state_i.setAccel(derivatives_p.col(2));
  state_i.setJerk(derivatives_p.col(3));

  return state_i;
}

// Given the control points, this function returns the associated traj and mt::PieceWisePol
void CPs2Traj(std::vector<Eigen::Vector3d> &qp, std::vector<double> &qy, Eigen::RowVectorXd &knots_p,
              Eigen::RowVectorXd &knots_y, std::vector<mt::state> &traj, int deg_p, int deg_y, double dc)
{
  assert(((knots_p.size() - 1) == (qp.size() - 1) + deg_p + 1) && "M=N+p+1 not satisfied");

  int num_seg = (knots_p.size() - 1) - 2 * deg_p;  // M-2*p

  // Stack the control points in matrices
  Eigen::Matrix<double, 3, -1> qp_matrix(3, qp.size());
  for (int i = 0; i < qp.size(); i++)
  {
    qp_matrix.col(i) = qp[i];
  }

  Eigen::Matrix<double, 1, -1> qy_matrix(1, qy.size());
  for (int i = 0; i < qy.size(); i++)
  {
    qy_matrix(0, i) = qy[i];
  }

  /////////////////////////////////////////////////////////////////////
  /// FILL ALL THE FIELDS OF TRAJ (BOTH POSITION AND YAW)
  /////////////////////////////////////////////////////////////////////

  // Construct now the B-Spline, see https://github.com/libigl/eigen/blob/master/unsupported/test/splines.cpp#L37
  Eigen::Spline<double, 3, Eigen::Dynamic> spline_p(knots_p, qp_matrix);
  Eigen::Spline<double, 1, Eigen::Dynamic> spline_y(knots_y, qy_matrix);

  // Note that t_min and t_max are the same for both yaw and position
  double t_min = knots_p.minCoeff();
  double t_max = knots_p.maxCoeff();

  // Clear and fill the trajectory
  traj.clear();

  for (double t = t_min; t <= t_max; t = t + dc)
  {
    // std::cout << std::setprecision(20) << "t= " << t << std::endl;
    // std::cout << std::setprecision(20) << "knots_p(0)= " << knots_p(0) << std::endl;
    Eigen::MatrixXd derivatives_p = spline_p.derivatives(t, 4);  // compute the derivatives up to that order
    Eigen::MatrixXd derivatives_y = spline_y.derivatives(t, 3);

    mt::state state_i;

    state_i.setPos(derivatives_p.col(0));  // First column
    state_i.setVel(derivatives_p.col(1));
    state_i.setAccel(derivatives_p.col(2));
    state_i.setJerk(derivatives_p.col(3));

    // state_i.printHorizontal();

    // if (isnan(state_i.pos.x()))
    // {
    //   abort();
    // }

    // std::cout << "derivatives_y= " << derivatives_y << std::endl;

    state_i.setYaw(derivatives_y(0, 0));
    state_i.setDYaw(derivatives_y(0, 1));
    state_i.setDDYaw(derivatives_y(0, 2));

    traj.push_back(state_i);
  }
}

void CPs2Traj(std::vector<Eigen::Vector2d> &qp, std::vector<double> &qy, Eigen::RowVectorXd &knots_p,
              Eigen::RowVectorXd &knots_y, std::vector<mt::state> &traj, int deg_p, int deg_y, double dc)
{
  assert(((knots_p.size() - 1) == (qp.size() - 1) + deg_p + 1) && "M=N+p+1 not satisfied");

  int num_seg = (knots_p.size() - 1) - 2 * deg_p;  // M-2*p

  // Stack the control points in matrices
  Eigen::Matrix<double, 2, -1> qp_matrix(2, qp.size());
  for (int i = 0; i < qp.size(); i++)
  {
    qp_matrix.col(i) = qp[i];
  }

  /////////////////////////////////////////////////////////////////////
  /// FILL ALL THE FIELDS OF TRAJ (BOTH POSITION AND YAW)
  /////////////////////////////////////////////////////////////////////

  // Construct now the B-Spline, see https://github.com/libigl/eigen/blob/master/unsupported/test/splines.cpp#L37
  Eigen::Spline<double, 2, Eigen::Dynamic> spline_p(knots_p, qp_matrix);

  // Note that t_min and t_max are the same for both yaw and position
  double t_min = knots_p.minCoeff();
  double t_max = knots_p.maxCoeff();

  // Clear and fill the trajectory
  traj.clear();

  for (double t = t_min; t <= t_max; t = t + dc)
  {
    Eigen::MatrixXd derivatives_p = spline_p.derivatives(t, 4);  // compute the derivatives up to that order
    mt::state state_i;
    state_i.setPos(derivatives_p.col(0)[0], derivatives_p.col(0)[1]); 
    state_i.setVel(derivatives_p.col(1)[0], derivatives_p.col(1)[1]);
    state_i.setAccel(derivatives_p.col(2)[0], derivatives_p.col(2)[1]);
    state_i.setJerk(derivatives_p.col(3)[0], derivatives_p.col(3)[1]);
    traj.push_back(state_i);
  }
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

Eigen::Spline3d findInterpolatingBsplineNormalized(const std::vector<double> &times,
                                                   const std::vector<Eigen::Vector3d> &positions)
{
  if (times.size() != positions.size())
  {
    std::cout << "times.size() should be == positions.size()" << std::endl;
    abort();
  }

  // check that times is  increasing
  for (int i = 0; i < (times.size() - 1); i++)
  {
    if (times[i + 1] < times[i])
    {
      std::cout << "times should be increasing" << std::endl;
      abort();
    }
  }

  if ((times.back() - times.front()) < 1e-7)
  {
    std::cout << "there is no time span in the vector times" << std::endl;
    abort();
  }

  // See example here: https://github.com/libigl/eigen/blob/master/unsupported/test/splines.cpp

  Eigen::MatrixXd points(3, positions.size());
  for (int i = 0; i < positions.size(); i++)
  {
    points.col(i) = positions[i];
  }

  Eigen::RowVectorXd knots_normalized(times.size());
  Eigen::RowVectorXd knots(times.size());

  for (int i = 0; i < times.size(); i++)
  {
    knots(i) = times[i];
    knots_normalized(i) = (times[i] - times[0]) / (times.back() - times.front());
  }

  Eigen::Spline3d spline_normalized = Eigen::SplineFitting<Eigen::Spline3d>::Interpolate(points, 3, knots_normalized);

  // Eigen::Spline3d spline(knots, spline_normalized.ctrls());

  // for (int i = 0; i < points.cols(); ++i)
  // {
  //   std::cout << "findInterpolatingBspline 6" << std::endl;

  //   std::cout << "knots(i)= " << knots(i) << std::endl;
  //   std::cout << "knots= " << knots << std::endl;
  //   std::cout << "spline.ctrls()= " << spline.ctrls() << std::endl;

  //   Eigen::Vector3d pt1 = spline_normalized(knots_normalized(i));
  //   std::cout << "pt1= " << pt1.transpose() << std::endl;

  //   // Eigen::Vector3d pt2 = spline(knots(i)); //note that spline(x) requires x to be in [0,1]

  //   // std::cout << "pt2= " << pt2.transpose() << std::endl;

  //   Eigen::Vector3d ref = points.col(i);
  //   std::cout << "norm= " << (pt1 - ref).norm() << std::endl;  // should be ~zero
  //   // std::cout << "norm= " << (pt2 - ref).norm() << std::endl;  // should be ~zero
  // }

  return spline_normalized;
}