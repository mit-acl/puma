/* ----------------------------------------------------------------------------
 * Copyright 2021, Jesus Tordesillas Torres, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Jesus Tordesillas, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

#include <casadi/casadi.hpp>

#include "solver_ipopt.hpp"
#include "termcolor.hpp"
#include "bspline_utils.hpp"
#include "ros/ros.h"

#include <unsupported/Eigen/Splines>
#include <iostream>
#include <list>
#include <random>
#include <iostream>
#include <vector>
#include <fstream>

#include <ros/package.h>

using namespace termcolor;

struct PrintSupresser
{
  PrintSupresser(){};
  ~PrintSupresser()
  {
    end();
  };
  void start()
  {
    std::cout.setstate(std::ios_base::failbit);  // https://stackoverflow.com/a/30185095
  }
  void end()
  {
    std::cout.clear();
  }
};

template <typename T>
int sgn(T val)
{
  return (T(0) < val) - (val < T(0));
}

std::vector<Eigen::Vector3d> casadiMatrix2StdVectorEigen3d(const casadi::DM &qp_casadi)
{
  std::vector<Eigen::Vector3d> qp;
  for (int i = 0; i < qp_casadi.columns(); i++)
  {
    qp.push_back(Eigen::Vector3d(double(qp_casadi(0, i)), double(qp_casadi(1, i)), double(qp_casadi(2, i))));
  }
  return qp;
}

std::vector<Eigen::Vector2d> casadiMatrix2StdVectorEigen2d(const casadi::DM &qp_casadi)
{
  std::vector<Eigen::Vector2d> qp;
  for (int i = 0; i < qp_casadi.columns(); i++)
  {
    qp.push_back(Eigen::Vector2d(double(qp_casadi(0, i)), double(qp_casadi(1, i))));
  }
  return qp;
}

std::vector<Eigen::Matrix<double, 9, 1>> casadiMatrix2StdVectorEigen9d(const casadi::DM &qp_casadi)
{
  std::vector<Eigen::Matrix<double, 9, 1>> qp;
  for (int i = 0; i < qp_casadi.columns(); i++)
  {
    Eigen::Matrix<double, 9, 1> tmp;
    tmp << double(qp_casadi(0, i)), double(qp_casadi(1, i)), double(qp_casadi(2, i)), double(qp_casadi(3, i)),
        double(qp_casadi(4, i)), double(qp_casadi(5, i)), double(qp_casadi(6, i)), double(qp_casadi(7, i)),
        double(qp_casadi(8, i));
    qp.push_back(tmp);
  }
  return qp;
}

std::vector<double> casadiMatrix2StdVectorDouble(const casadi::DM &qy_casadi)
{
  return static_cast<std::vector<double>>(qy_casadi);
}

casadi::DM stdVectorEigen3d2CasadiMatrix(const std::vector<Eigen::Vector3d> &qp)
{
  casadi::DM casadi_matrix(2, qp.size());  // TODO: do this just once?
  for (int i = 0; i < casadi_matrix.columns(); i++)
  {
    casadi_matrix(0, i) = qp[i].x();
    casadi_matrix(1, i) = qp[i].y();
  }
  return casadi_matrix;
}

casadi::DM stdVectorEigen2d2CasadiMatrix(const std::vector<Eigen::Vector2d> &qp)
{
  casadi::DM casadi_matrix(2, qp.size());  // TODO: do this just once?
  for (int i = 0; i < casadi_matrix.columns(); i++)
  {
    casadi_matrix(0, i) = qp[i].x();
    casadi_matrix(1, i) = qp[i].y();
  }
  return casadi_matrix;
}

casadi::DM stdVectorDouble2CasadiRowVector(const std::vector<double> &qy)
{
  casadi::DM casadi_matrix(1, qy.size());
  for (int i = 0; i < casadi_matrix.columns(); i++)
  {
    casadi_matrix(0, i) = qy[i];
  }
  return casadi_matrix;
}

casadi::DM eigenXd2CasadiMatrix(const Eigen::Matrix<double, Eigen::Dynamic, 1> &data)
{
  casadi::DM casadi_matrix(data.rows(), 1);

  for (int i = 0; i < data.rows(); i++)
  {
    casadi_matrix(i, 0) = data(i);
  }

  return casadi_matrix;
}

std::vector<Eigen::Matrix<double, 2, 1>> StdVectorEigen3d2StdVectorEigen2d(const std::vector<Eigen::Vector3d> &qp)
{
  std::vector<Eigen::Matrix<double, 2, 1>> qp2d;
  for (int i = 0; i < qp.size(); i++)
  {
    qp2d.push_back(Eigen::Matrix<double, 2, 1>(qp[i].x(), qp[i].y()));
  }
  return qp2d;
}

///////////////////////

Fitter::Fitter(const int fitter_num_samples)
{
  std::string folder = ros::package::getPath("puma") + "/matlab/casadi_generated_files/";
  cf_fit2d_ = casadi::Function::load(folder + "acslam_fit2d.casadi");
  fitter_num_samples_ = fitter_num_samples;
}

Fitter::~Fitter()
{
}

std::vector<Eigen::Vector2d> Fitter::fit(std::vector<Eigen::Vector3d> &samples)
{
  verify(samples.size() == fitter_num_samples_, "The number of samples needs to be equal to "
                                                "fitter_num_samples");

  // Fit a spline to those samples
  std::map<std::string, casadi::DM> map_arg;
  map_arg["samples"] = stdVectorEigen3d2CasadiMatrix(samples);
  std::map<std::string, casadi::DM> result = cf_fit2d_(map_arg);
  std::vector<Eigen::Vector2d> ctrl_pts = casadiMatrix2StdVectorEigen2d(result["result"]);

  return ctrl_pts;
}

SolverIpopt::SolverIpopt(const mt::parameters &par)
{

  // Parameters initialization
  par_ = par;
  si::splineParam sp_tmp(par_.deg_pos, par_.num_seg);
  sp_ = sp_tmp;

  // Basis initialization
  mt::basisConverter basis_converter;
  if (par_.basis == "MINVO")
  {
    basis_ = MINVO;
    M_pos_bs2basis_ = basis_converter.getMinvoDeg3Converters(par_.num_seg);
    M_vel_bs2basis_ = basis_converter.getMinvoDeg2Converters(par_.num_seg);
  }
  else if (par_.basis == "BEZIER")
  {
    basis_ = BEZIER;
    M_pos_bs2basis_ = basis_converter.getBezierDeg3Converters(par_.num_seg);
    M_vel_bs2basis_ = basis_converter.getBezierDeg2Converters(par_.num_seg);
  }
  else if (par_.basis == "B_SPLINE")
  {
    basis_ = B_SPLINE;
    M_pos_bs2basis_ = basis_converter.getBSplineDeg3Converters(par_.num_seg);
    M_vel_bs2basis_ = basis_converter.getBSplineDeg2Converters(par_.num_seg);
  }
  else
  {
    std::cout << red << "Basis " << par_.basis << " not implemented yet" << reset << std::endl;
    std::cout << red << "============================================" << reset << std::endl;
    abort();
  }

  // Casadi files initialization
  // TODO: if C++14, use std::make_unique instead
  separator_solver_ptr_ = std::unique_ptr<separator::Separator>(new separator::Separator());
  octopusSolver_ptr_ = std::unique_ptr<OctopusSearch>(new OctopusSearch(par_.basis, par_.num_seg, par_.deg_pos, par_.alpha_shrink));
  std::cout << bold << "SolverIpopt, reading .casadi files..." << reset << std::endl;
  std::string folder = ros::package::getPath("puma") + "/matlab/casadi_generated_files/";
  std::fstream myfile(folder + "acslam_index_instruction.txt", std::ios_base::in);
  myfile >> index_instruction_;
  cf_compute_cost_ = casadi::Function::load(folder + "acslam_compute_cost.casadi");
  cf_op_ = casadi::Function::load(folder + "acslam_op.casadi");
  cf_compute_dyn_limits_constraints_violation_ = casadi::Function::load(folder + "acslam_compute_dyn_limits_constraints_violation.casadi");
  cf_compute_trans_and_yaw_dyn_limits_constraints_violatoin_ = casadi::Function::load(folder + "acslam_compute_trans_dyn_limits_constraints_violation.casadi");
}

SolverIpopt::~SolverIpopt()
{
}

void SolverIpopt::getPlanes(std::vector<Hyperplane3D> &planes)
{
  planes = planes_;
}

bool SolverIpopt::isInCollision(mt::state state, double t)
{
  for (const auto &obstacle_i : obstacles_for_opt_)
  {
    Eigen::RowVectorXd knots_p =
        constructKnotsClampedUniformSpline(t_init_, t_final_guess_, par_.fitter_deg_pos, par_.fitter_num_seg);

    mt::state state_obs = getStatePosSplineT(obstacle_i.ctrl_pts, knots_p, sp_.p, t);

    Eigen::Array<double, 3, 1> distance = (state_obs.pos - state.pos).array().abs();
    Eigen::Vector3d delta = obstacle_i.bbox_inflated / 2.0;

    if ((distance < delta.array()).all())
    {
      std::cout << "state_obs.pos= " << state_obs.pos.transpose() << std::endl;
      std::cout << "state.pos= " << state.pos.transpose() << std::endl;
      std::cout << "distance= " << distance.transpose() << std::endl;
      std::cout << "delta= " << delta.transpose() << std::endl;

      return true;
    }
  }
  return false;
}

void SolverIpopt::setObstaclesForOpt(const std::vector<mt::obstacleForOpt> &obstacles_for_opt)
{

  obstacles_for_opt_ = obstacles_for_opt;

  ////// Set the hulls for use in Octopus Search
  hulls_.clear();

  Eigen::RowVectorXd knots_p =
      constructKnotsClampedUniformSpline(t_init_, t_final_guess_, par_.fitter_deg_pos, par_.fitter_num_seg);

  double deltaT = (t_final_guess_ - t_init_) / (1.0 * par_.num_seg);  // num_seg is the number of intervals

  for (const auto &obstacle_i : obstacles_for_opt_)
  {

    // right now we are treating obstacle and agent the same way
    // maybe in the future we want to treat them differently

    VertexesObstacle vertexes_obstacle_i;

    for (int j = 0; j < par_.num_seg; j++)
    {
      std::vector<double> times =
          linspace(t_init_ + j * deltaT, t_init_ + (j + 1) * deltaT, par_.disc_pts_per_interval_oct_search);
      VertexesInterval vertexes_interval_j(3, 8 * times.size());  // For each sample, there 1 are 8 vertexes

      for (int k = 0; k < times.size(); k++)
      {
        mt::state state = getStatePosSplineT(obstacle_i.ctrl_pts, knots_p, sp_.p, times[k]);
        Eigen::Vector3d delta;
        delta = obstacle_i.bbox_inflated / 2.0;

        // clang-format off
        vertexes_interval_j.col(8*k)=     (Eigen::Vector3d(state.pos.x() + delta.x(), state.pos.y() + delta.y(), state.pos.z() + delta.z()));
        vertexes_interval_j.col(8*k+1)=   (Eigen::Vector3d(state.pos.x() + delta.x(), state.pos.y() - delta.y(), state.pos.z() - delta.z()));
        vertexes_interval_j.col(8*k+2)=   (Eigen::Vector3d(state.pos.x() + delta.x(), state.pos.y() + delta.y(), state.pos.z() - delta.z()));
        vertexes_interval_j.col(8*k+3)=   (Eigen::Vector3d(state.pos.x() + delta.x(), state.pos.y() - delta.y(), state.pos.z() + delta.z()));
        vertexes_interval_j.col(8*k+4)=   (Eigen::Vector3d(state.pos.x() - delta.x(), state.pos.y() - delta.y(), state.pos.z() - delta.z()));
        vertexes_interval_j.col(8*k+5)=   (Eigen::Vector3d(state.pos.x() - delta.x(), state.pos.y() + delta.y(), state.pos.z() + delta.z()));
        vertexes_interval_j.col(8*k+6)=   (Eigen::Vector3d(state.pos.x() - delta.x(), state.pos.y() + delta.y(), state.pos.z() - delta.z()));
        vertexes_interval_j.col(8*k+7)=   (Eigen::Vector3d(state.pos.x() - delta.x(), state.pos.y() - delta.y(), state.pos.z() + delta.z()));
        // clang-format on

      }

      vertexes_obstacle_i.push_back(vertexes_interval_j);
    }

    hulls_.push_back(vertexes_obstacle_i);

  }

  num_of_obst_ = hulls_.size();
  num_of_normals_ = par_.num_seg * num_of_obst_;
}

casadi::DM SolverIpopt::eigen2casadi(const Eigen::Vector3d &a)
{
  casadi::DM b = casadi::DM::zeros(3, 1);
  b(0, 0) = a(0);
  b(1, 0) = a(1);
  b(2, 0) = a(2);
  return b;
};

// Note that t_final will be updated in case the saturation in deltaT has had effect
bool SolverIpopt::setInitStateFinalStateInitTFinalT(mt::state initial_state, mt::state final_state, double t_init,
                                                    double &t_final)
{
  ///////////////////////////
  Eigen::Vector3d p0 = initial_state.pos;
  Eigen::Vector3d v0 = initial_state.vel;
  Eigen::Vector3d a0 = initial_state.accel;

  initial_state_ = initial_state;
  final_state_ = final_state;

  initial_state_.yaw = wrapFromMPitoPi(initial_state_.yaw);
  final_state_.yaw = wrapFromMPitoPi(final_state_.yaw);

  /// Now shift final_state_.yaw  so that the difference wrt initial_state_.yaw is <=pi

  double previous_phi = initial_state_.yaw;
  double phi_i = final_state_.yaw;
  double difference = previous_phi - phi_i;

  double phi_i_f = phi_i + floor(difference / (2 * M_PI)) * 2 * M_PI;
  double phi_i_c = phi_i + ceil(difference / (2 * M_PI)) * 2 * M_PI;

  final_state_.yaw = (fabs(previous_phi - phi_i_f) < fabs(previous_phi - phi_i_c)) ? phi_i_f : phi_i_c;

  /// Just for debugging
  if (fabs(initial_state_.yaw - final_state_.yaw) > M_PI)
  {
    std::cout << red << bold << "This diff must be <= pi" << reset << std::endl;
    abort();
  }

  double deltaT = (t_final - t_init) / (1.0 * (sp_.M - 2 * sp_.p - 1 + 1));

  t_final = t_init + (1.0 * (sp_.M - 2 * sp_.p - 1 + 1)) * deltaT;

  t_init_ = t_init;
  t_final_guess_ = t_final;

  // std::cout << "total_time_guess= " << t_final_guess_ - t_init_ << std::endl;

  return true;
}

std::vector<si::solOrGuess> SolverIpopt::getBestSolutions()
{
  return solutions_;
}

si::solOrGuess SolverIpopt::getBestSolution()
{
  double min_cost = std::numeric_limits<double>::max();
  int argmin = -1;
  for (int i = 0; i < solutions_.size(); i++)
  {
    if (solutions_[i].solver_succeeded && (solutions_[i].cost < min_cost))
    {
      min_cost = solutions_[i].cost;
      argmin = i;
    }
  }

  if (argmin < 0)
  {
    std::cout << bold << red << "Aborting: You called fillTrajBestSololutionAndGetIt after optimize() was false"
              << reset << std::endl;
    abort();
  }

  return solutions_[argmin];
}

si::solOrGuess SolverIpopt::fillTrajBestSolutionAndGetIt()
{

  si::solOrGuess best_solution = getBestSolution();
  best_solution.fillTraj(par_.dc);

  // Force last vel and jerk =final_state_ (which it's not guaranteed because of the discretization with par_.dc)
  best_solution.traj.back().vel = final_state_.vel;
  best_solution.traj.back().accel = final_state_.accel;
  best_solution.traj.back().jerk = Eigen::Vector3d::Zero();

  return best_solution;
}

std::vector<si::solOrGuess> SolverIpopt::getGuesses()
{
  return guesses_;
}

double SolverIpopt::computeCost(si::solOrGuess sol_or_guess)
{
  std::map<std::string, casadi::DM> map_arguments = getMapConstantArguments();

  casadi::DM matrix_qp = stdVectorEigen2d2CasadiMatrix(sol_or_guess.qp);
  casadi::DM matrix_qy = stdVectorDouble2CasadiRowVector(sol_or_guess.qy);

  map_arguments["alpha"] = sol_or_guess.getTotalTime();
  map_arguments["pCPs"] = matrix_qp;
  map_arguments["yCPs"] = matrix_qy;

  for (int i = 0; i < par_.num_max_of_obst; i++)
  {
    map_arguments["obs_" + std::to_string(i) + "_ctrl_pts"] =
        stdVectorEigen2d2CasadiMatrix(obstacles_for_opt_[i].ctrl_pts);
    map_arguments["obs_" + std::to_string(i) + "_bbox_inflated"] =
        eigenXd2CasadiMatrix(obstacles_for_opt_[i].bbox_inflated);
  }

  // map_arguments["all_nd"] = all_nd;                                 // Only appears in the constraints

  std::map<std::string, casadi::DM> result = cf_compute_cost_(map_arguments);

  double cost = double(result["total_cost"]);
  return cost;
}

double SolverIpopt::computeDynLimitsConstraintsViolation(si::solOrGuess sol_or_guess)
{
  std::map<std::string, casadi::DM> map_arguments = getMapConstantArguments();

  casadi::DM matrix_qp = stdVectorEigen2d2CasadiMatrix(sol_or_guess.qp);
  casadi::DM matrix_qy = stdVectorDouble2CasadiRowVector(sol_or_guess.qy);

  map_arguments["alpha"] = sol_or_guess.getTotalTime();
  map_arguments["pCPs"] = matrix_qp;
  map_arguments["yCPs"] = matrix_qy;
  // map_arguments["all_nd"] = all_nd;  //Does not appear in the dyn. limits constraints

  std::map<std::string, casadi::DM> result = cf_compute_dyn_limits_constraints_violation_(map_arguments);

  std::vector<double> dyn_limits_constraints_violation = casadiMatrix2StdVectorDouble(result["violation"]);

  double violation = std::accumulate(
      dyn_limits_constraints_violation.begin(), dyn_limits_constraints_violation.end(),
      decltype(dyn_limits_constraints_violation)::value_type(0));  // https://stackoverflow.com/a/3221813/6057617

  return violation;
}

std::map<std::string, casadi::DM> SolverIpopt::getMapConstantArguments()
{
  // Conversion DM <--> Eigen:  https://github.com/casadi/casadi/issues/2563
  // return 2D vector
  auto eigen2std = [](Eigen::Vector3d &v) { return std::vector<double>{ v.x(), v.y() }; };

  std::map<std::string, casadi::DM> map_arguments;
  map_arguments["Ra"] = par_.Ra;
  map_arguments["p0"] = eigen2std(initial_state_.pos);
  map_arguments["v0"] = eigen2std(initial_state_.vel);
  map_arguments["a0"] = eigen2std(initial_state_.accel);
  map_arguments["pf"] = eigen2std(final_state_.pos);
  map_arguments["vf"] = eigen2std(final_state_.vel);
  map_arguments["af"] = eigen2std(final_state_.accel);
  map_arguments["v_max"] = eigen2std(par_.v_max);
  map_arguments["a_max"] = eigen2std(par_.a_max);
  map_arguments["j_max"] = eigen2std(par_.j_max);
  map_arguments["x_lim"] = std::vector<double>{ par_.x_min, par_.x_max };
  map_arguments["y_lim"] = std::vector<double>{ par_.y_min, par_.y_max };

  std::cout << "add obstacles" << std::endl;
  // convert obstacles_for_opt_.size() to int
  int tmp = static_cast<int>(obstacles_for_opt_.size());
  int obstacle_size = std::min(tmp, par_.num_max_of_obst);
  for (int i = 0; i < obstacle_size; i++)
  {
    map_arguments["obs_" + std::to_string(i) + "_ctrl_pts"] =
        stdVectorEigen2d2CasadiMatrix(obstacles_for_opt_[i].ctrl_pts);
    map_arguments["obs_" + std::to_string(i) + "_bbox_inflated"] =
        eigenXd2CasadiMatrix(obstacles_for_opt_[i].bbox_inflated.head(2));
  }

  map_arguments["c_pos_smooth"] = par_.c_pos_smooth;
  map_arguments["c_final_pos"] = par_.c_final_pos;
  map_arguments["c_total_time"] = par_.c_total_time;
  return map_arguments;
}

void SolverIpopt::getOptTime(double &opt_time)
{
  opt_time = opt_timer_.getMsSaved();
}

bool SolverIpopt::optimize(bool supress_all_prints)
{
  info_last_opt_ = "";

  PrintSupresser print_supresser;
  if (supress_all_prints)
  {
    print_supresser.start();
  }
  std::cout << "in SolverIpopt::optimize" << std::endl;

  std::cout << "initial_state= " << std::endl;
  initial_state_.printHorizontal();

  std::cout << "final_state= " << std::endl;
  final_state_.printHorizontal();

  std::cout << "obstacles = " << std::endl;
  for (auto &tmp : obstacles_for_opt_)
  {
    tmp.printInfo();
  }

  std::cout << "v_max= " << par_.v_max.transpose() << std::endl;
  std::cout << "a_max= " << par_.a_max.transpose() << std::endl;
  std::cout << "j_max= " << par_.j_max.transpose() << std::endl;
  std::vector<os::solution> p_guesses;

  // reset some stuff
  solutions_.clear();

  bool guess_found = generateAStarGuess(p_guesses);  // obtain p_guesses
  if (guess_found == false)
  {
    info_last_opt_ = "OS: " + std::to_string(p_guesses.size());
    std::cout << bold << red << "No guess for pos found" << reset << std::endl;
    return false;
  }

  int max_num_of_planes = par_.num_max_of_obst * par_.num_seg;
  if ((p_guesses[0].n.size() > max_num_of_planes))
  {
    info_last_opt_ = "The casadi function does not support so many planes";
    std::cout << red << bold << info_last_opt_ << reset << std::endl;
    std::cout << red << bold << "you have " << num_of_obst_ << "*" << par_.num_seg << "=" << p_guesses[0].n.size() << " planes" << std::endl;
    std::cout << red << bold << "and max is  " << par_.num_max_of_obst << "*" << par_.num_seg << "=" << max_num_of_planes << " planes" << std::endl;
    return false;
  }

  //
  // CASADI (getmapConstantArguments will get obstacle info, etc)
  //

  std::cout << "solve CASADI" << std::endl;
  std::map<std::string, casadi::DM> map_arguments = getMapConstantArguments();

  double alpha_guess = (t_final_guess_ - t_init_);
  if (alpha_guess < par_.lower_bound_alpha && !par_.use_panther_star) {
    alpha_guess = par_.lower_bound_alpha;
  }
  map_arguments["alpha"] = alpha_guess;  // Initial guess for alpha
  std::cout << "alpha_guess= " << alpha_guess << std::endl;

  //
  // SOLVE AN OPIMIZATION FOR EACH OF THE GUESSES FOUND
  //

  std::vector<si::solOrGuess> solutions;
  std::vector<si::solOrGuess> guesses;
  std::vector<std::string> opt_statuses;

  // #pragma omp parallel for
  for (auto p_guess : p_guesses)
  {
    static casadi::DM all_nd(3, max_num_of_planes);
    all_nd = casadi::DM::zeros(3, max_num_of_planes);
    for (int i = 0; i < p_guess.n.size(); i++)
    {
      // The optimized curve is on the side n'x+d <= -1
      // The obstacle is on the side n'x+d >= 1
      all_nd(0, i) = p_guess.n[i].x();
      all_nd(1, i) = p_guess.n[i].y();
      all_nd(2, i) = p_guess.d[i];
    }

    map_arguments["all_nd"] = all_nd;

    ///////////////// GUESS FOR POSITION CONTROL POINTS

    casadi::DM matrix_qp_guess = stdVectorEigen3d2CasadiMatrix(p_guess.qp);

    map_arguments["pCPs"] = matrix_qp_guess;

    si::solOrGuess guess;
    guess.deg_p = par_.deg_pos;
    guess.is_guess = true;
    guess.qp = StdVectorEigen3d2StdVectorEigen2d(p_guess.qp);
    guess.knots_p = constructKnotsClampedUniformSpline(t_init_, t_final_guess_, sp_.p, sp_.num_seg);

    std::map<std::string, casadi::DM> result;
    opt_timer_.tic();
    result = cf_op_(map_arguments);  // from Casadi
    opt_timer_.toc();

    std::string optimstatus = std::string(cf_op_.instruction_MX(index_instruction_).which_function().stats(1)["return_status"]);

    si::solOrGuess solution;
    solution.is_guess = false;
    solution.cost = double(result["total_cost"]);
    solution.dyn_lim_violation = 0.0;         // Because it's feasible
    solution.obst_avoidance_violation = 0.0;  // Because it's feasible
    solution.deg_p = par_.deg_pos;
    solution.deg_y = par_.deg_yaw;

    // hack (TODO): sometimes the total time is very small (and final position is very close to initial position)
    if (double(result["alpha"]) < 1e-4)
    {
      optimstatus = "The alpha found is too small";
    }

    opt_statuses.push_back(optimstatus);
    std::cout << "optimstatus= " << optimstatus << std::endl;

    bool success_opt;

    // See names here:
    // https://github.com/casadi/casadi/blob/fadc86444f3c7ab824dc3f2d91d4c0cfe7f9dad5/casadi/interfaces/ipopt/ipopt_interface.cpp#L368
    if (optimstatus == "Solve_Succeeded")  //|| optimstatus == "Solved_To_Acceptable_Level"
    {
      std::cout << green << "IPOPT found a solution" << reset << std::endl;
      success_opt = true;
      solution.qp = casadiMatrix2StdVectorEigen2d(result["pCPs"]);
      solution.knots_p = getKnotsSolution(guess.knots_p, alpha_guess, double(result["alpha"]));
      solution.alpha = double(result["alpha"]);
      double total_time_solution = (solution.knots_p(solution.knots_p.cols() - 1) - solution.knots_p(0));
      
      if (total_time_solution > (par_.fitter_total_time + 1e-4))
      {
        std::cout << yellow << bold << "WARNING: total_time_solution>par_.fitter_total_time (visibility/obstacle samples are not taken in t>par_.fitter_total_time)" << reset << std::endl;
        std::cout << "total_time_solution= " << total_time_solution << std::endl;
        std::cout << "par_.fitter_total_time= " << par_.fitter_total_time << std::endl;
        std::cout << yellow << bold << "Increase fitter.total_time (or decrease Ra)" << reset << std::endl;
        // abort();  // Debugging
      }
    }
    else
    {
      std::cout << red << "IPOPT failed to find a solution" << reset << std::endl;
      success_opt = false;
    }

    solution.solver_succeeded = success_opt;
    solutions.push_back(solution);
    guesses.push_back(guess);
  }

  solutions = getOnlySucceededAndDifferent(solutions);

  struct
  {
    bool operator()(si::solOrGuess &a, si::solOrGuess &b) const
    {
      return a.cost < b.cost;
    }
  } customLess;
  std::sort(solutions.begin(), solutions.end(), customLess);  // sort solutions from lowest to highest cost
  solutions_ = solutions;
  guesses_ = guesses;
  std::cout << bold << red << solutions_.size() << "/" << par_.num_of_trajs_per_replan << " solutions found" << reset << std::endl;
  info_last_opt_ = "OS: " + std::to_string(p_guesses.size()) + ", Ipopt: " + std::to_string(solutions_.size()) + " --> ";
  
  for (auto &opt_status : opt_statuses)
  {
    std::string tmp = opt_status;
    if (opt_status == "Solve_Succeeded")
    {
      tmp = "S";
    }

    if (opt_status == "Infeasible_Problem_Detected")
    {
      tmp = "I";
    }

    info_last_opt_ = info_last_opt_ + "|" + tmp;
  }

  // NO SOLUTION FOUND
  if (solutions_.size() == 0)
  {
    std::cout << "Ipopt found zero solutions" << std::endl;
    return false;
  }
  // TOO MANY SOLUTIONS: RETAIN the ones with lowest cost (the first ones)
  else if (solutions_.size() > par_.num_of_trajs_per_replan)
  {
    int elements_to_delete = solutions_.size() - par_.num_of_trajs_per_replan;
    solutions_.erase(solutions_.end() - elements_to_delete, solutions_.end());
  }
  // TOO FEW SOLUTIONS: DUPLICATE SOME//TODO: any better option?
  else
  {
    si::solOrGuess dummy_solution = getBestSolution();  // Copy the best solution found
    dummy_solution.solver_succeeded = false;
    dummy_solution.is_repeated = true;
    while (solutions_.size() < par_.num_of_trajs_per_replan)
    {
      solutions_.push_back(dummy_solution);
    }
  }

  return true;

}

std::string SolverIpopt::getInfoLastOpt()
{
  return info_last_opt_;
}

std::vector<si::solOrGuess> SolverIpopt::getOnlySucceeded(std::vector<si::solOrGuess> solutions)
{
  std::vector<si::solOrGuess> solutions_succeeded;
  for (auto &solution : solutions)
  {
    if (solution.solver_succeeded)
    {
      solutions_succeeded.push_back(solution);
    }
  }
  return solutions_succeeded;
}

std::vector<si::solOrGuess> SolverIpopt::getOnlySucceededAndDifferent(std::vector<si::solOrGuess> solutions)
{
  std::vector<si::solOrGuess> solutions_succeeded = getOnlySucceeded(solutions);

  std::vector<int> indexes_to_delete;

  int num_sol_succeded =
      int(solutions_succeeded.size());  // This must be an int, see https://stackoverflow.com/a/65645023/6057617

  for (int i = 0; i < (num_sol_succeded - 1); i++)
  {
    // std::cout << "i= " << i << std::endl;

    // solutions_succeeded[i].printInfo();
    for (int j = i + 1; j < num_sol_succeded; j++)
    {
      // std::cout << "j= " << j << std::endl;
      si::solOrGuess sol1 = solutions_succeeded[i];
      si::solOrGuess sol2 = solutions_succeeded[j];
      bool are_different = false;
      for (int k = 0; k < sol1.qp.size(); k++)
      {
        double distance_cpk = (sol1.qp[k] - sol2.qp[k]).norm();
        if (distance_cpk > 1e-3)  // Sufficiently different in the translational space. TODO:
                                  // include yaw
                                  // + time?
        {
          // std::cout << "distance cpk=" << distance_cpk << std::endl;
          // std::cout << "sol1.qp[k]= " << sol1.qp[k].transpose() << std::endl;
          // std::cout << "sol2.qp[k]= " << sol2.qp[k].transpose() << std::endl;
          are_different = true;
          break;
        }
      }
      // std::cout << "are_different= " << are_different << std::endl;
      // if reached this point, the two solutions are the same ones
      if (are_different == false)
      {
        // std::cout << "Pushing " << j << std::endl;
        indexes_to_delete.push_back(j);
        // std::cout << "Going to delete j!" << std::endl;
      }
      // std::cout << "here" << std::endl;
    }
  }

  // https://stackoverflow.com/a/1041939
  sort(indexes_to_delete.begin(), indexes_to_delete.end());
  indexes_to_delete.erase(unique(indexes_to_delete.begin(), indexes_to_delete.end()), indexes_to_delete.end());
  /////

  // std::cout << "end of all loops" << std::endl;
  // std::cout << "indexes_to_delete.size()= " << indexes_to_delete.size() << std::endl;

  for (int i = indexes_to_delete.size() - 1; i >= 0; i--)
  {
    solutions_succeeded.erase(solutions_succeeded.begin() + indexes_to_delete[i]);
  }

  // std::cout << "different solutions_succeeded.size()=" << solutions_succeeded.size() << std::endl;

  return solutions_succeeded;
}

int SolverIpopt::numSolutionsSucceeded()
{
  int result = 0;
  for (auto &solution : solutions_)
  {
    if (solution.solver_succeeded)
    {
      result += 1;
    }
  }
  return result;
}

bool SolverIpopt::anySolutionSucceeded()
{
  return (numSolutionsSucceeded() > 0);
}

std::vector<double> SolverIpopt::yawCPsToGoToFinalYaw(double deltaT)
{
  std::vector<double> qy;

  double y0 = initial_state_.yaw;
  double yf = final_state_.yaw;
  double ydot0 = initial_state_.dyaw;
  int p = par_.deg_yaw;

  qy.clear();
  qy.push_back(y0);
  qy.push_back(y0 + deltaT * ydot0 / (double(p)));  // y0 and ydot0 fix the second control point

  int num_cps_yaw = par_.num_seg + p;

  for (int i = 0; i < (num_cps_yaw - 3); i++)
  {
    double v_needed = p * (yf - qy.back()) / (p * deltaT);

    saturate(v_needed, -par_.ydot_max, par_.ydot_max);  // Make sure it's within the limits

    double next_qy = qy.back() + (p * deltaT) * v_needed / (double(p));

    qy.push_back(next_qy);
  }

  qy.push_back(qy.back());  // TODO: HERE I'M ASSUMMING FINAL YAW VELOCITY=0 (i.e., final_state_.dyaw==0)

  return qy;
}

Eigen::RowVectorXd SolverIpopt::getKnotsSolution(const Eigen::RowVectorXd &knots_guess, const double alpha_guess,
                                                 const double alpha_solution)
{
  int num_knots = knots_guess.cols();

  // std::cout << "knots_guess= " << knots_guess << std::endl;

  Eigen::RowVectorXd shift = knots_guess(0) * Eigen::RowVectorXd::Ones(1, num_knots);

  // std::cout << "shift= " << shift << std::endl;

  Eigen::RowVectorXd knots_solution = (knots_guess - shift) * (alpha_solution / alpha_guess) + shift;

  // std::cout << "knots_solution= " << knots_solution << std::endl;

  return knots_solution;
}
// void SolverIpopt::getSolution(mt::PieceWisePol &solution)
// {
//   solution = pwp_solution_;
// }

void SolverIpopt::fillPlanesFromNDQ(const std::vector<Eigen::Vector3d> &n, const std::vector<double> &d,
                                    const std::vector<Eigen::Vector3d> &q)
{
  planes_.clear();

  for (int obst_index = 0; obst_index < num_of_obst_; obst_index++)
  {
    for (int i = 0; i < par_.num_seg; i++)
    {
      int ip = obst_index * par_.num_seg + i;  // index plane
      Eigen::Vector3d centroid_hull;
      findCentroidHull(hulls_[obst_index][i], centroid_hull);

      Eigen::Vector3d point_in_plane;

      Eigen::Matrix<double, 3, 4> Qmv, Qbs;  // minvo. each column contains a MINVO control point
      Qbs.col(0) = q[i];
      Qbs.col(1) = q[i + 1];
      Qbs.col(2) = q[i + 2];
      Qbs.col(3) = q[i + 3];

      transformPosBSpline2otherBasis(Qbs, Qmv, i);

      Eigen::Vector3d centroid_cps = Qmv.rowwise().mean();

      // the colors refer to the second figure of
      // https://github.com/mit-acl/separator/tree/06c0ddc6e2f11dbfc5b6083c2ea31b23fd4fa9d1

      // Equation of the red planes is n'x+d == 1
      // Convert here to equation [A B C]'x+D ==0
      double A = n[ip].x();
      double B = n[ip].y();
      double C = n[ip].z();
      double D = d[ip] - 1;

      /////////////////// OPTION 1: point_in_plane = intersection between line  centroid_cps --> centroid_hull
      // bool intersects = getIntersectionWithPlane(centroid_cps, centroid_hull, Eigen::Vector4d(A, B, C, D),
      //                                            point_in_plane);  // result saved in point_in_plane

      //////////////////////////

      /////////////////// OPTION 2: point_in_plane = intersection between line  centroid_cps --> closest_vertex
      double dist_min = std::numeric_limits<double>::max();  // delta_min will contain the minimum distance between
                                                             // the centroid_cps and the vertexes of the obstacle
      int index_closest_vertex = 0;
      for (int j = 0; j < hulls_[obst_index][i].cols(); j++)
      {
        Eigen::Vector3d vertex = hulls_[obst_index][i].col(j);

        double distance_to_vertex = (centroid_cps - vertex).norm();
        if (distance_to_vertex < dist_min)
        {
          dist_min = distance_to_vertex;
          index_closest_vertex = j;
        }
      }

      Eigen::Vector3d closest_vertex = hulls_[obst_index][i].col(index_closest_vertex);

      bool intersects = getIntersectionWithPlane(centroid_cps, closest_vertex, Eigen::Vector4d(A, B, C, D),
                                                 point_in_plane);  // result saved in point_in_plane

      //////////////////////////

      if (intersects == false)
      {
        // TODO: this msg is printed sometimes in Multi-Agent simulations. Find out why
        std::cout << red << "There is no intersection, this should never happen (TODO)" << reset << std::endl;
        continue;  // abort();
      }

      Hyperplane3D plane(point_in_plane, n[i]);
      planes_.push_back(plane);
    }
  }
}

// returns 1 if there is an intersection between the segment P1-P2 and the plane given by coeff=[A B C D]
// (Ax+By+Cz+D==0)  returns 0 if there is no intersection.
// The intersection point is saved in "intersection"
bool SolverIpopt::getIntersectionWithPlane(const Eigen::Vector3d &P1, const Eigen::Vector3d &P2,
                                           const Eigen::Vector4d &coeff, Eigen::Vector3d &intersection)
{
  double A = coeff[0];
  double B = coeff[1];
  double C = coeff[2];
  double D = coeff[3];
  // http://www.ambrsoft.com/TrigoCalc/Plan3D/PlaneLineIntersection_.htm
  double x1 = P1[0];
  double a = (P2[0] - P1[0]);
  double y1 = P1[1];
  double b = (P2[1] - P1[1]);
  double z1 = P1[2];
  double c = (P2[2] - P1[2]);
  double t = -(A * x1 + B * y1 + C * z1 + D) / (A * a + B * b + C * c);

  (intersection)[0] = x1 + a * t;
  (intersection)[1] = y1 + b * t;
  (intersection)[2] = z1 + c * t;

  bool result = (t < 0 || t > 1) ? false : true;  // False if the intersection is with the line P1-P2, not with the
                                                  // segment P1 - P2

  return result;
}

//  casadi::DM all_nd(casadi::Sparsity::dense(4, max_num_of_planes));
// casadi::DM::rand(4, 0);

// std::string getPathName(const std::string &s)
// {
//   char sep = '/';

// #ifdef _WIN32
//   sep = '\\';
// #endif

//   size_t i = s.rfind(sep, s.length());
//   if (i != std::string::npos)
//   {
//     return (s.substr(0, i));
//   }

//   return ("");
// }