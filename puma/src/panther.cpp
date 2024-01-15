/* ----------------------------------------------------------------------------
 * Copyright 2021, Jesus Tordesillas Torres, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Jesus Tordesillas, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

#include <Eigen/StdVector>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <stdlib.h>
#include <ros/package.h>
#include "panther.hpp"
#include "timer.hpp"
#include "termcolor.hpp"
#include "bspline_utils.hpp"
// Needed to call the student
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
// #include "exprtk.hpp"

using namespace termcolor;

// Uncomment the type of timer you want:
// typedef ROSTimer MyTimer;
// typedef ROSWallTimer MyTimer;
typedef PANTHER_timers::Timer MyTimer;

//
// ------------------------------------------------------------------------------------------------------
//

Panther::Panther(mt::parameters par) : par_(par)
{
  drone_status_ == DroneStatus::GOAL_REACHED;
  G_.pos << 0, 0;
  G_term_.pos << 0, 0;
  mtx_initial_cond.lock();
  stateA_.setZero();
  mtx_initial_cond.unlock();
  changeDroneStatus(DroneStatus::GOAL_REACHED);
  resetInitialization();
  solver_ = new SolverIpopt(par_);                //, log_ptr_
  fitter_ = new Fitter(par_.fitter_num_samples);  //, log_ptr_
  separator_solver_ = new separator::Separator();
}

//
// ------------------------------------------------------------------------------------------------------
//

Panther::~Panther()
{
}

//
// ------------------------------------------------------------------------------------------------------
//

void Panther::getNumObstAndNumOtherAgents(const std::vector<mt::obstacleForOpt>& obstacles_for_opt, int& num_obst,
                                          int& num_oa)
{
  num_obst = 0;
  num_oa = 0;
  for (auto obst : obstacles_for_opt)
  {
    if (obst.is_agent == false)
    {
      num_obst++;
    }
    else
    {
      num_oa++;
    }
  }
}

//
// ------------------------------------------------------------------------------------------------------
//

void Panther::dynTraj2dynTrajCompiled(const mt::dynTraj& traj, mt::dynTrajCompiled& traj_compiled)
{
  if (traj.use_pwp_field == true)
  {
    traj_compiled.pwp_mean = traj.pwp_mean;
    traj_compiled.pwp_var = traj.pwp_var;
    traj_compiled.is_static =
        ((traj.pwp_mean.eval(0.0) - traj.pwp_mean.eval(1e30)).norm() < 1e-5);  // TODO: Improve this
  }
  else
  {
    mtx_t_.lock();

    // typedef exprtk::symbol_table<double> symbol_table_t;
    // typedef exprtk::expression<double> expression_t;
    // typedef exprtk::parser<double> parser_t;

    // Compile the mean
    for (auto function_i : traj.s_mean)
    {
      MathEvaluator engine;
      engine.add_var("t", t_);
      engine.compile(function_i);
      traj_compiled.s_mean.push_back(engine);
    }

    // Compile the variance
    for (auto function_i : traj.s_var)
    {
      MathEvaluator engine;
      engine.add_var("t", t_);
      engine.compile(function_i);
      traj_compiled.s_var.push_back(engine);
    }

    mtx_t_.unlock();

    traj_compiled.is_static =
        (traj.s_mean[0].find("t") == std::string::npos) &&  // there is no dependence on t in the coordinate x
        (traj.s_mean[1].find("t") == std::string::npos) &&  // there is no dependence on t in the coordinate y
        (traj.s_mean[2].find("t") == std::string::npos);    // there is no dependence on t in the coordinate z
  }

  traj_compiled.use_pwp_field = traj.use_pwp_field;
  traj_compiled.is_agent = traj.is_agent;
  traj_compiled.bbox = traj.bbox;
  traj_compiled.id = traj.id;
  traj_compiled.time_received = traj.time_received;  // ros::Time::now().toSec();
  traj_compiled.is_committed = traj.is_committed;
}

//
// ------------------------------------------------------------------------------------------------------
//

// Note that this function is here because I need t_ for this evaluation
Eigen::Vector3d Panther::evalMeanDynTrajCompiled(const mt::dynTrajCompiled& traj, double t)
{
  Eigen::Vector3d tmp;
  if (traj.use_pwp_field == true)
  {
    tmp = traj.pwp_mean.eval(t);
  }
  else
  {
    mtx_t_.lock();
    t_ = t;
    tmp << traj.s_mean[0].value(),  ////////////////
        traj.s_mean[1].value(),     ////////////////
        traj.s_mean[2].value();     ////////////////

    mtx_t_.unlock();
  }
  return tmp;
}

//
// ------------------------------------------------------------------------------------------------------
//

// Note that this function is here because it needs t_ for this evaluation
Eigen::Vector3d Panther::evalVarDynTrajCompiled(const mt::dynTrajCompiled& traj, double t)
{
  Eigen::Vector3d tmp;

  if (traj.use_pwp_field == true)
  {
    tmp = traj.pwp_var.eval(t);
  }
  else
  {
    mtx_t_.lock();
    t_ = t;
    tmp << traj.s_var[0].value(),  ////////////////
        traj.s_var[1].value(),     ////////////////
        traj.s_var[2].value();     ////////////////

    mtx_t_.unlock();
  }
  return tmp;
}

//
// ------------------------------------------------------------------------------------------------------
//

void Panther::removeOldTrajectories()
{
  double time_now = ros::Time::now().toSec();
  std::vector<int> ids_to_remove;

  mtx_trajs_.lock();

  for (int index_traj = 0; index_traj < trajs_.size(); index_traj++)
  {
    if ((time_now - trajs_[index_traj].time_received) > par_.max_seconds_keeping_traj)
    {
      // std::cout << "remove traj " << std::endl;
      // std::cout << "time_now " << time_now << std::endl;
      // std::cout << "trajs_[" << index_traj << "].time_received " << trajs_[index_traj].time_received << std::endl; 
      ids_to_remove.push_back(trajs_[index_traj].id);
    }
  }

  for (auto id : ids_to_remove)
  {
    // ROS_WARN_STREAM("Removing " << id);
    trajs_.erase(
        std::remove_if(trajs_.begin(), trajs_.end(), [&](mt::dynTrajCompiled const& traj) { return traj.id == id; }),
        trajs_.end());
  }

  mtx_trajs_.unlock();
}

//
// ------------------------------------------------------------------------------------------------------
//
// Note that we need to compile the trajectories inside panther.cpp because t_ is in panther.hpp
void Panther::updateTrajObstacles(mt::dynTraj traj)
{
  MyTimer tmp_t(true);

  //
  // used for Recheck
  //

  if (started_check_ == true && traj.is_agent == true)
  {
    have_received_trajectories_while_checking_ = true;
  }

  //
  // Get traj_compiled
  //

  mt::dynTrajCompiled traj_compiled;
  dynTraj2dynTrajCompiled(traj, traj_compiled);

  //
  // Add traj_compiled to trajs_
  //

  mtx_trajs_.lock();

  //
  // If the trajectory is not committed, then we add it to trajs_ regardless of whether it already exists or not
  //

  if (!traj_compiled.is_committed)
  {
    trajs_.push_back(traj_compiled);
  }
  else
  {
    std::vector<mt::dynTrajCompiled>::iterator obs_ptr =
        std::find_if(trajs_.begin(), trajs_.end(),
                     [=](const mt::dynTrajCompiled& traj_compiled) { return traj_compiled.id == traj.id; });

    bool exists_in_local_map = (obs_ptr != std::end(trajs_));

    if (exists_in_local_map)
    {  // if that object already exists, substitute its trajectory
      *obs_ptr = traj_compiled;
    }
    else
    {  // if it doesn't exist, add it to the local map
      trajs_.push_back(traj_compiled);
      // ROS_WARN_STREAM("Adding " << traj_compiled.id);
    }
  }

  mtx_trajs_.unlock();

  have_received_trajectories_while_checking_ = false;
}

bool Panther::IsTranslating()
{
  return (drone_status_ == DroneStatus::GOAL_SEEN || drone_status_ == DroneStatus::TRAVELING);
}

//
// ------------------------------------------------------------------------------------------------------
//

std::vector<mt::obstacleForOpt> Panther::getObstaclesForOpt(double t_start, double t_end)
{
  std::vector<mt::obstacleForOpt> obstacles_for_opt;

  double delta = (t_end - t_start) / par_.fitter_num_samples;

  for (int i = 0; i < trajs_.size(); i++)
  {
    mt::dynTrajCompiled traj = trajs_[i];
    mt::obstacleForOpt obstacle_for_opt;

    // Take future samples of the trajectory
    std::vector<Eigen::Vector3d> samples;

    for (int k = 0; k < par_.fitter_num_samples; k++)
    {
      double tk = t_start + k * delta;
      Eigen::Vector3d pos_k = evalMeanDynTrajCompiled(traj, tk);
      samples.push_back(pos_k);
    }

    obstacle_for_opt.ctrl_pts = fitter_->fit(samples);
    Eigen::Vector3d bbox_inflated = traj.bbox + par_.drone_bbox / 2;
    obstacle_for_opt.bbox_inflated = bbox_inflated;
    obstacle_for_opt.is_agent = traj.is_agent;
    obstacles_for_opt.push_back(obstacle_for_opt);
  }

  // std::cout << "obstacles_for_opt.size() " << obstacles_for_opt.size() << std::endl;

  return obstacles_for_opt;
}

//
// ------------------------------------------------------------------------------------------------------
//

void Panther::setTerminalGoal(mt::state& term_goal)
{
  mtx_G_term.lock();
  G_term_ = term_goal;
  mtx_G_term.unlock();

  if (state_initialized_ == true)  // because I need plan_size()>=1
  {
    doStuffTermGoal();
  }
  else
  {
    need_to_do_stuff_term_goal_ = true;  // will be done in updateState();
  }
}

//
// ------------------------------------------------------------------------------------------------------
//

void Panther::getG(mt::state& G)
{
  G = G_;
}

//
// ------------------------------------------------------------------------------------------------------
//

void Panther::getState(mt::state& data)
{
  mtx_state_.lock();
  data = state_;
  mtx_state_.unlock();
}

//
// ------------------------------------------------------------------------------------------------------
//

void Panther::getG_term(mt::state& data)
{
  mtx_G_term.lock();
  data = G_term_;  // Local copy of the terminal terminal goal
  mtx_G_term.unlock();
}

//
// ------------------------------------------------------------------------------------------------------
//

void Panther::updateState(mt::state data)
{
  state_ = data;

  mtx_plan_.lock();
  if (state_initialized_ == false || plan_.size() == 1)
  {
    plan_.clear();  // (actually not needed because done in resetInitialization()
    mt::state tmp;
    tmp.pos = data.pos;
    tmp.yaw = data.yaw;
    plan_.push_back(tmp);
  }
  
  if (plan_.size() == 1)
  {
    // std::cout << "plan_.size() == 1" << std::endl;
    plan_.clear();  // (actually not needed because done in resetInitialization()
    mt::state tmp;
    tmp.pos = data.pos;
    tmp.yaw = data.yaw;
    plan_.push_back(tmp);
  }
  
  mtx_plan_.unlock();

  state_initialized_ = true;

  if (need_to_do_stuff_term_goal_)
  {
    // std::cout << "DOING STUFF TERM GOAL -----------" << std::endl;
    doStuffTermGoal();
    need_to_do_stuff_term_goal_ = false;
  }
}

//
// ------------------------------------------------------------------------------------------------------
//
// This function needs to be called once the state has been initialized
void Panther::doStuffTermGoal()
{
  // get G_term
  mtx_G_term.lock();
  G_.pos = G_term_.pos;
  mtx_G_term.unlock();

  terminal_goal_initialized_ = true;
  is_new_g_term_ = true;
  changeDroneStatus(DroneStatus::TRAVELING);
}

//
// ------------------------------------------------------------------------------------------------------
//

bool Panther::initializedAllExceptPlanner()
{
  if (!state_initialized_ || !terminal_goal_initialized_)
  {
    /*    std::cout << "state_initialized_= " << state_initialized_ << std::endl;
        std::cout << "terminal_goal_initialized_= " << terminal_goal_initialized_ << std::endl;*/
    return false;
  }
  return true;
}

//
// ------------------------------------------------------------------------------------------------------
//

bool Panther::initializedStateAndTermGoal()
{
  if (!state_initialized_ || !terminal_goal_initialized_)
  {
    return false;
  }
  return true;
}

//
// ------------------------------------------------------------------------------------------------------
//

bool Panther::initialized()
{
  if (!state_initialized_ || !terminal_goal_initialized_ || !planner_initialized_)
  {
    /*    std::cout << "state_initialized_= " << state_initialized_ << std::endl;
        std::cout << "terminal_goal_initialized_= " << terminal_goal_initialized_ << std::endl;
        std::cout << "planner_initialized_= " << planner_initialized_ << std::endl;*/
    return false;
  }
  return true;
}

//
// ------------------------------------------------------------------------------------------------------
//

bool Panther::isReplanningNeeded()
{
  if (initializedStateAndTermGoal() == false)
  {
    std::cout << "Not replanning because state or terminal goal not initialized" << std::endl;
    return false;  // Note that log is not modified --> will keep its default values
  }

  if (par_.static_planning == true)
  {
    std::cout << "Not replanning because static_planning==true" << std::endl;
    return true;
  }

  //////////////////////////////////////////////////////////////////////////
  mtx_G_term.lock();
  mt::state G_term = G_term_;  // Local copy of the terminal terminal goal
  mtx_G_term.unlock();

  mtx_state_.lock();
  mt::state current_state = state_;
  mtx_state_.unlock();

  //
  // Check if goal is seen and distance to goal is less than goal_seen_radius
  //

  //
  // Check if we have seen the goal in the last replan
  //

  mtx_plan_.lock();
  double dist_last_plan_to_goal = (G_term.pos - plan_.back().pos).norm();
  mtx_plan_.unlock();

  mtx_plan_.lock();
  double dist_to_goal = (G_term.pos - current_state.pos).norm();
  mtx_plan_.unlock();

  if (dist_last_plan_to_goal < par_.goal_radius && dist_to_goal < par_.goal_seen_radius)
  {
    changeDroneStatus(DroneStatus::GOAL_SEEN);
    std::cout << "Status changed to GOAL_SEEN!" << std::endl;
    std::cout << "dist_to_goal= " << dist_to_goal << std::endl;
    std::cout << "goal_seen_radius= " << par_.goal_seen_radius << std::endl;
    exists_previous_pwp_ = false;
  }

  // Check if we have reached the goal
  if (dist_to_goal < par_.goal_radius)
  {
    changeDroneStatus(DroneStatus::GOAL_REACHED);
    exists_previous_pwp_ = false;
  }

  // Don't plan if drone is not traveling
  if (drone_status_ == DroneStatus::GOAL_REACHED || (drone_status_ == DroneStatus::GOAL_SEEN)) {
    return false;
  }

  return true;
}

//
// ------------------------------------------------------------------------------------------------------
//

void Panther::pubObstacleEdge(mt::Edges& edges_obstacles_out, const Eigen::Affine3d& c_T_b,
                              const Eigen::Affine3d& w_T_b)
{
  //
  // Get edges_obstacles
  //

  double t_start = ros::Time::now().toSec();
  double t_final = t_start + par_.obstacle_visualization_duration;

  // remove old trajectories
  removeOldTrajectories();

  ConvexHullsOfCurves hulls = convexHullsOfCurvesForObstacleEdge(t_start, t_final, c_T_b, w_T_b);
  edges_obstacles_out = cu::vectorGCALPol2edges(hulls);
}

//
// ------------------------------------------------------------------------------------------------------
//

void Panther::adjustObstaclesForOptimization(std::vector<mt::obstacleForOpt>& obstacles_for_opt)
{
  //
  // If obstacles_for_opt is too small, then we need to add some dummy obstacles, which is the copy of the last obstacle
  //

  if (obstacles_for_opt.size() < par_.num_max_of_obst)
  {
    std::cout << bold << "Too few obstacles. duplicate and add the last obstacles" << std::endl;
    for (int i = obstacles_for_opt.size(); i < par_.num_max_of_obst; i++)
    {
      mt::obstacleForOpt dummy_obst;
      dummy_obst = obstacles_for_opt.back();
      obstacles_for_opt.push_back(dummy_obst);
    }
  }

  //
  // If obstacles_for_opt is too large, then we need to delete some of them
  //

  if (obstacles_for_opt.size() > par_.num_max_of_obst)
  {
    std::cout << red << bold << "Too many obstacles. Run Matlab again with a higher num_max_of_obst" << reset
              << std::endl;
    for (int i = obstacles_for_opt.size(); i > par_.num_max_of_obst; i--)
    {
      obstacles_for_opt.pop_back();
    }
  }

  //
  // If using student, and there's no other agent, then we need to add a dummy other agent (otherwise, LSTM for other
  // agent will complain)
  //

  int num_obst, num_oa;
  getNumObstAndNumOtherAgents(obstacles_for_opt, num_obst, num_oa);

  if (par_.use_student == true && num_oa == 0)
  {
    std::cout << bold << "No other agent. Add a dummy other agent" << std::endl;
    mt::obstacleForOpt& dummy_oa = obstacles_for_opt.back();  // this could change obstacle into other agent, but it's
                                                              // better than using a completely dummy other agent
    dummy_oa.is_agent = true;
  }

  if (par_.use_student == true && num_obst == 0)
  {
    std::cout << bold << "No other agent. Add a dummy other agent" << std::endl;
    mt::obstacleForOpt& dummy_oa = obstacles_for_opt.front();  // this could change obstacle into other agent, but it's
                                                              // better than using a completely dummy other agent
    dummy_oa.is_agent = false;
  }
}

//
// ------------------------------------------------------------------------------------------------------
//

bool Panther::replan(si::solOrGuess& best_solution, mt::log& log, int& k_index_end)
{

  //
  // Check if we need to replan
  //

  mtx_G_term.lock();
  mt::state G_term = G_term_;  // Local copy of the terminal terminal goal
  mtx_G_term.unlock();

  if (isReplanningNeeded() == false)
  {
    std::cout << "No replanning needed" << std::endl;
    return false;
  }

  std::cout << bold << on_white << "**********************IN REPLAN CB*******************" << reset << std::endl;

  removeOldTrajectories();

  //////////////////////////////////////////////////////////////////////////
  ///////////////////////// Select mt::state A /////////////////////////////
  //////////////////////////////////////////////////////////////////////////

  mt::state A;
  int k_index;

  // If k_index_end=0, then A = plan_.back() = plan_[plan_.size() - 1]

  mtx_plan_.lock();

  // saturate(deltaT_, par_.lower_bound_runtime_snlopt / par_.dc, par_.upper_bound_runtime_snlopt / par_.dc);

  deltaT_ = par_.replanning_lookahead_time / par_.dc;  // Added October 18, 2021

  k_index_end = std::max((int)(plan_.size() - deltaT_), 0);

  if (plan_.size() < 5)
  {
    k_index_end = 0;
  }

  if (par_.static_planning)  // plan from the same position all the time
  {
    k_index_end = plan_.size() - 1;
  }

  k_index = plan_.size() - 1 - k_index_end;
  A = plan_.get(k_index);
  mtx_plan_.unlock();

  //////////////////////////////////////////////////////////////////////////
  ///////////////////////// Get point G ////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////

  double distA2TermGoal = (G_term.pos - A.pos).norm();
  double ra = std::min((distA2TermGoal - 0.001), par_.Ra);  // radius of the sphere S
  mt::state G;
  G.pos = A.pos + ra * (G_term.pos - A.pos).normalized();

  double time_now = ros::Time::now().toSec();  // shouldn't have ros dependency here
  double t_start = k_index * par_.dc + time_now;

  mtx_trajs_.lock();
  std::vector<mt::obstacleForOpt> obstacles_for_opt = getObstaclesForOpt(t_start, t_start + par_.fitter_total_time);  // basically converts trajs_ to obstacles_for_opt
  mtx_trajs_.unlock();

  // setobstacles for optimization
  solver_->setObstaclesForOpt(obstacles_for_opt);

  // Verify that obstacles_for_opt.size() is less than par_.num_max_of_obst
  verify(obstacles_for_opt.size() <= par_.num_max_of_obst, "obstacles_for_opt.size() should be less than par_.num_max_of_obst");

  // get t_final
  double time_allocated = getMinTimeDoubleIntegrator3D(A.pos, A.vel, G.pos, G.vel, par_.v_max, par_.a_max);
  double t_final = t_start + par_.factor_alloc * time_allocated;

  // Optimization

  solver_->setFocusOnObstacle(true);
  solver_->par_.c_fov = par_.c_fov;
  solver_->par_.c_final_yaw = par_.c_final_yaw;
  solver_->par_.c_yaw_smooth = par_.c_yaw_smooth;

  bool correctInitialCond = solver_->setInitStateFinalStateInitTFinalT(A, G, t_start, t_final);

  if (correctInitialCond == false) {
    return false;
  }

  // Solve optimization! 
  bool result = solver_->optimize(false);
  if (result == false) {
    return false;
  }
  best_solution = solver_->fillTrajBestSolutionAndGetIt();
  total_replannings_++;


  solutions_found_++;

  //
  // OTHER STUFF
  //

  planner_initialized_ = true;

  // success
  return true;
}

//
// ------------------------------------------------------------------------------------------------------
//

bool Panther::addTrajToPlan(const int& k_index_end, mt::log& log, const si::solOrGuess& best_solution, mt::trajectory& traj)
{

  // mutex lock
  mtx_plan_.lock();

  // get the plan size
  int plan_size = plan_.size();

  // check if we already passed A
  if ((plan_size - 1 - k_index_end) < 0)
  {
    std::cout << "plan_size= " << plan_size << std::endl;
    std::cout << "k_index_end= " << k_index_end << std::endl;
    mtx_plan_.unlock();
    return false;
  }
  else
  {
    plan_.erase(plan_.end() - k_index_end - 1, plan_.end());  // this deletes also the initial condition...
    for (auto& state : best_solution.traj)  //... which is included in best_solution.traj[0]
    {
      plan_.push_back(state);
    }
  }

  // mutex unlock
  mtx_plan_.unlock();

  // fill traj for visualization
  traj = plan_.toStdVector();

  //
  return true;
}

//
// --------------------------------------------------------------------------------------------------------------
//

// Check and Recheck (used in MADER)
bool Panther::safetyCheck(mt::PieceWisePol& pwp)
{
  mtx_trajs_.lock();
  started_check_ = true;
  bool result = true;

  //
  // Check
  //

  for (auto traj : trajs_)
  {
    if (traj.time_received > time_init_opt_ && traj.is_agent == true)
    {
      if (trajsAndPwpAreInCollision(traj, pwp, pwp.times.front(), pwp.times.back()))
      {
        mtx_trajs_.unlock();
        ROS_ERROR_STREAM("[Check] Traj collides with " << traj.id);
        result = false;  // will have to redo the optimization
        return result;
      }
    }
  }

  mtx_trajs_.unlock();

  //
  // Recheck
  //

  // and now do another check in case I've received anything while I was checking.
  if (have_received_trajectories_while_checking_ == true)
  {
    ROS_ERROR_STREAM("[Recheck] Recvd traj while checking ");
    result = false;
  }

  started_check_ = false;

  return result;
}

//
// ------------------------------------------------------------------------------------------------------
//

// Check step used in Delay Check
bool Panther::check(mt::PieceWisePol& pwp)
{
  bool result = true;
  mtx_trajs_.lock();

  //
  // Check
  //

  for (auto traj : trajs_)
  {
    if (traj.time_received > time_init_opt_ && traj.is_agent == true)
    {
      if (trajsAndPwpAreInCollision(traj, pwp, pwp.times.front(), pwp.times.back()))
      {
        mtx_trajs_.unlock();
        ROS_ERROR_STREAM("[Check] Traj collides with " << traj.id);
        result = false;  // will have to redo the optimization
        return result;
      }
    }
  }

  mtx_trajs_.unlock();

  return result;
}

//
// --------------------------------------------------------------------------------------------------------------
//

// Delay Check
bool Panther::delayCheck(mt::PieceWisePol& pwp)
{
  auto start = std::chrono::system_clock::now();
  auto current_time = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = current_time - start;

  // delay check for par_.delaycheck_time seconds
  while (elapsed_seconds.count() < par_.delaycheck_time)
  {
    if (!check(pwp))
    {
      ROS_ERROR_STREAM("[Delay Check] Traj collides");
      return false;
    }
    current_time = std::chrono::system_clock::now();
    elapsed_seconds = current_time - start;
  }

  // one last time to make sure the all the traj received in delaycheck_time have been checked
  if (!check(pwp))
  {
    ROS_ERROR_STREAM("[Delay Check] Traj collides");
    return false;
  }

  return true;
}

//
// ------------------------------------------------------------------------------------------------------
//

bool Panther::trajsAndPwpAreInCollision(mt::dynTrajCompiled& traj, mt::PieceWisePol& pwp, const double& t_start,
                                        const double& t_end)
{
  Eigen::Vector3d n_i;
  double d_i;
  double deltaT = (t_end - t_start) / (1.0 * par_.num_seg);
  for (int i = 0; i < par_.num_seg; i++)  // for each interval
  {
    // This is my trajectory (no inflation)
    std::vector<Eigen::Vector3d> pointsA =
        vertexesOfInterval(pwp, t_start + i * deltaT, t_start + (i + 1) * deltaT, Eigen::Vector3d::Zero());

    // This is the trajectory of the other agent/obstacle
    std::vector<Eigen::Vector3d> pointsB = vertexesOfInterval(traj, t_start + i * deltaT, t_start + (i + 1) * deltaT);

    if (separator_solver_->solveModel(n_i, d_i, pointsA, pointsB) == false)
    {
      return true;  // There is not a solution --> they collide
    }
  }

  // if reached this point, they don't collide
  return false;
}

//
// ------------------------------------------------------------------------------------------------------
//

std::vector<Eigen::Vector3d> Panther::vertexesOfInterval(mt::dynTrajCompiled& traj, double t_start, double t_end)
{
  Eigen::Vector3d delta = Eigen::Vector3d::Zero();
  Eigen::Vector3d drone_boundarybox = par_.drone_bbox;

  if (traj.is_agent == false)
  {
    std::vector<Eigen::Vector3d> points;
    delta = traj.bbox + drone_boundarybox / 2.0;

    // Will always have a sample at the beginning of the interval, and another at the end.
    for (double t = t_start;                           /////////////
         (t < t_end) ||                                /////////////
         ((t > t_end) && ((t - t_end) < par_.gamma));  /////// This is to ensure we have a sample a the end
         t = t + par_.gamma)
    {
      mtx_t_.lock();
      t_ = std::min(t, t_end);  // this min only has effect on the last sample

      double x = traj.s_mean[0].value();
      double y = traj.s_mean[1].value();
      double z = traj.s_mean[2].value();
      mtx_t_.unlock();

      //"Minkowski sum along the trajectory: box centered on the trajectory"
      points.push_back(Eigen::Vector3d(x + delta.x(), y + delta.y(), z + delta.z()));
      points.push_back(Eigen::Vector3d(x + delta.x(), y - delta.y(), z - delta.z()));
      points.push_back(Eigen::Vector3d(x + delta.x(), y + delta.y(), z - delta.z()));
      points.push_back(Eigen::Vector3d(x + delta.x(), y - delta.y(), z + delta.z()));
      points.push_back(Eigen::Vector3d(x - delta.x(), y - delta.y(), z - delta.z()));
      points.push_back(Eigen::Vector3d(x - delta.x(), y + delta.y(), z + delta.z()));
      points.push_back(Eigen::Vector3d(x - delta.x(), y + delta.y(), z - delta.z()));
      points.push_back(Eigen::Vector3d(x - delta.x(), y - delta.y(), z + delta.z()));
    }

    return points;
  }
  else
  {  // is an agent --> use the pwp field
    delta = traj.bbox + drone_boundarybox / 2.0;
    return vertexesOfInterval(traj.pwp_mean, t_start, t_end, delta);
  }
}

//
// ------------------------------------------------------------------------------------------------------
//

std::vector<Eigen::Vector3d> Panther::vertexesOfInterval(mt::PieceWisePol& pwp, double t_start, double t_end, const Eigen::Vector3d& delta)
{
  std::vector<Eigen::Vector3d> points;

  std::vector<double>::iterator low = std::lower_bound(pwp.times.begin(), pwp.times.end(), t_start);
  std::vector<double>::iterator up = std::upper_bound(pwp.times.begin(), pwp.times.end(), t_end);

  // // Example: times=[1 2 3 4 5 6 7]
  // // t_start=1.5;
  // // t_end=5.5
  // // then low points to "2" (low - pwp.times.begin() is 1)
  // // and up points to "6" (up - pwp.times.begin() is 5)

  int index_first_interval = low - pwp.times.begin() - 1;  // index of the interval [1,2]
  int index_last_interval = up - pwp.times.begin() - 1;    // index of the interval [5,6]

  saturate(index_first_interval, 0, (int)(pwp.all_coeff_x.size() - 1));
  saturate(index_last_interval, 0, (int)(pwp.all_coeff_x.size() - 1));

  Eigen::Matrix<double, 3, 4> P;
  Eigen::Matrix<double, 3, 4> V;

  mt::basisConverter basis_converter;
  Eigen::Matrix<double, 4, 4> A_rest_pos_basis_inverse = basis_converter.getArestMinvoDeg3().inverse();

  // push all the complete intervals
  for (int i = index_first_interval; i <= index_last_interval; i++)
  {

    // get t
    double t = pwp.times[i];
    P.row(0) = pwp.all_coeff_x[i];
    P.row(1) = pwp.all_coeff_y[i];
    P.row(2) = pwp.all_coeff_z[i];

    V = P * A_rest_pos_basis_inverse;

    for (int j = 0; j < V.cols(); j++)
    {
      double x = V(0, j);
      double y = V(1, j);
      double z = V(2, j);  //[x,y,z] is the point

      if (delta.norm() < 1e-6)
      {  // no inflation
        points.push_back(Eigen::Vector3d(x, y, z));
      }
      else
      {
        // points.push_back(Eigen::Vector3d(V(1, j), V(2, j), V(3, j)));  // x,y,z

        if (j == V.cols() - 1)
        {
          points.push_back(Eigen::Vector3d(x + delta.x(), y + delta.y(), z + delta.z()));
          points.push_back(Eigen::Vector3d(x + delta.x(), y - delta.y(), z - delta.z()));
          points.push_back(Eigen::Vector3d(x + delta.x(), y + delta.y(), z - delta.z()));
          points.push_back(Eigen::Vector3d(x + delta.x(), y - delta.y(), z + delta.z()));
          points.push_back(Eigen::Vector3d(x - delta.x(), y - delta.y(), z - delta.z()));
          points.push_back(Eigen::Vector3d(x - delta.x(), y + delta.y(), z + delta.z()));
          points.push_back(Eigen::Vector3d(x - delta.x(), y + delta.y(), z - delta.z()));
          points.push_back(Eigen::Vector3d(x - delta.x(), y - delta.y(), z + delta.z()));
        }
        else
        {
          points.push_back(Eigen::Vector3d(x, y, z));
        }
      }
    }
  }

  return points;
}

//
// ------------------------------------------------------------------------------------------------------
//

void Panther::printInfo(si::solOrGuess& best_solution, int n_safe_trajs)
{
  std::cout << std::setprecision(8)
            << "CostTimeResults: [Cost, obst_avoidance_violation, dyn_lim_violation, total_time_ms, n_safe_trajs]= "
            << best_solution.cost << " " << best_solution.obst_avoidance_violation << " "
            << best_solution.dyn_lim_violation << " " << log_ptr_->tim_total_replan.elapsedSoFarMs() << " "
            << n_safe_trajs << std::endl;
}

//
// ------------------------------------------------------------------------------------------------------
//

void Panther::logAndTimeReplan(const std::string& info, const bool& success, mt::log& log)
{
  log_ptr_->info_replan = info;
  log_ptr_->tim_total_replan.toc();
  log_ptr_->success_replanning = success;

  double total_time_ms = log_ptr_->tim_total_replan.getMsSaved();

  mtx_offsets.lock();
  if (success == false)
  {
    std::cout << bold << red << log_ptr_->info_replan << reset << std::endl;
    int states_last_replan = ceil(total_time_ms / (par_.dc * 1000));  // Number of states that
                                                                      // would have been needed for
                                                                      // the last replan
    deltaT_ = std::max(par_.factor_alpha * states_last_replan, 1.0);
    deltaT_ = std::min(1.0 * deltaT_, 2.0 / par_.dc);
  }
  else
  {
    int states_last_replan = ceil(total_time_ms / (par_.dc * 1000));  // Number of states that
                                                                      // would have been needed for
                                                                      // the last replan
    deltaT_ = std::max(par_.factor_alpha * states_last_replan, 1.0);
  }
  mtx_offsets.unlock();

  log = (*log_ptr_);
}

//
// ------------------------------------------------------------------------------------------------------
//

void Panther::resetInitialization()
{
  planner_initialized_ = false;
  state_initialized_ = false;

  terminal_goal_initialized_ = false;
  plan_.clear();
}

//
// ------------------------------------------------------------------------------------------------------
//

void Panther::convertAgentStateToCameraState(const mt::state& agent_state, mt::state& camera_state)
{ 
  camera_state = agent_state;
  // camera_state.yaw = agent_state.yaw - M_PI / 2; //TODO: hacky. proper way to do this to use drone2camera transform to get camera yaw
  camera_state.yaw = agent_state.yaw; //TODO: hacky. proper way to do this to use drone2camera transform to get camera yaw
}

//
// ------------------------------------------------------------------------------------------------------
//

void Panther::yaw(double diff, mt::state& next_goal)
{
  saturate(diff, -par_.dc * par_.ydot_max, par_.dc * par_.ydot_max);
  double dyaw_not_filtered;
  double alpha_filter_dyaw = 0.1;

  dyaw_not_filtered = copysign(1, diff) * par_.ydot_max;

  dyaw_filtered_ = (1 - alpha_filter_dyaw) * dyaw_not_filtered + alpha_filter_dyaw * dyaw_filtered_;
  next_goal.dyaw = dyaw_filtered_;
  next_goal.yaw = previous_yaw_ + dyaw_filtered_ * par_.dc;
  previous_yaw_ = next_goal.yaw;
  // std::cout << "next_goal.yaw " << next_goal.yaw << std::endl;
}

void Panther::getDesiredYaw(mt::state& next_goal)
{
  mt::state current_state;
  getState(current_state);
  // next_goal = yawing_start_state_;

  // double desired_yaw = atan2(G_term_.pos[1] - yawing_start_state_.pos[1], G_term_.pos[0] - yawing_start_state_.pos[0]);
  double desired_yaw = atan2(G_term_.pos[1] - current_state.pos[1], G_term_.pos[0] - current_state.pos[0]);

  mt::state camera_state;
  convertAgentStateToCameraState(current_state, camera_state);
  double diff = desired_yaw - camera_state.yaw;

  angle_wrap(diff);

  yaw(diff, next_goal);

  if (fabs(diff) < 0.04)
  {
    std::cout << bold << "YAWING IS DONE!!" << std::endl;
    changeDroneStatus(DroneStatus::TRAVELING);
  }
}

bool Panther::getNextGoal(mt::state& next_goal)
{
  if (initializedStateAndTermGoal() == false)  // || (drone_status_ == DroneStatus::GOAL_REACHED && plan_.size() == 1))
                                               // TODO: if included this part commented out, the last state (which is
                                               // the one that has zero accel) will never get published
  {
    // std::cout << "plan_.size() ==" << plan_.size() << std::endl;
    // std::cout << "plan_.content[0] ==" << std::endl;
    // plan_.content[0].print();
    return false;
  }

  mtx_goals.lock();
  mtx_plan_.lock();

  next_goal.setZero();
  next_goal = plan_.front();
  previous_yaw_ = next_goal.yaw;

  if (plan_.size() > 1)
  {
    if (par_.static_planning == false)
    {
      plan_.pop_front();
    }
  }

  mtx_goals.unlock();
  mtx_plan_.unlock();
  return true;
}

//
// ------------------------------------------------------------------------------------------------------
//

// Debugging functions
void Panther::changeDroneStatus(int new_status)
{
  if (new_status == drone_status_)
  {
    return;
  }

  std::cout << "Changing DroneStatus from ";
  switch (drone_status_)
  {
    case DroneStatus::TRAVELING:
      std::cout << bold << "TRAVELING" << reset;
      break;
    case DroneStatus::GOAL_SEEN:
      std::cout << bold << "GOAL_SEEN" << reset;
      break;
    case DroneStatus::GOAL_REACHED:
      std::cout << bold << "GOAL_REACHED" << reset;
      break;
  }
  std::cout << " to ";

  switch (new_status)
  {
    case DroneStatus::TRAVELING:
      std::cout << bold << "TRAVELING" << reset;
      break;
    case DroneStatus::GOAL_SEEN:
      std::cout << bold << "GOAL_SEEN" << reset;
      break;
    case DroneStatus::GOAL_REACHED:
      std::cout << bold << "GOAL_REACHED" << reset;
      break;
  }

  std::cout << std::endl;

  drone_status_ = new_status;
}

//
// ------------------------------------------------------------------------------------------------------
//

void Panther::printDroneStatus()
{
  switch (drone_status_)
  {
    case DroneStatus::TRAVELING:
      std::cout << bold << "status_=TRAVELING" << reset << std::endl;
      break;
    case DroneStatus::GOAL_SEEN:
      std::cout << bold << "status_=GOAL_SEEN" << reset << std::endl;
      break;
    case DroneStatus::GOAL_REACHED:
      std::cout << bold << "status_=GOAL_REACHED" << reset << std::endl;
      break;
  }
}

//
// ------------------------------------------------------------------------------------------------------
//

// Given the control points, this function returns the associated traj and mt::PieceWisePol
void Panther::convertsolOrGuess2pwp(mt::PieceWisePol& pwp_p, si::solOrGuess& solorguess, double dc)
{
  std::vector<Eigen::Vector2d> qp = solorguess.qp;
  std::vector<double> qy = solorguess.qy;
  Eigen::RowVectorXd knots_p = solorguess.knots_p;

  // Right now we use this function only for publishOwnTraj() and it doesn't matter yaw, so we don't have yaw pwp
  // param_pp is degree of position polynomial (p is degree of polynomial (usually p = 3))
  int param_pp = solorguess.deg_p;
  // param_pp is degree of yaw polynomial (p is degree of polynomial (usually p = 2))
  int param_py = solorguess.deg_y;

  assert(((knots_p.size() - 1) == (qp.size() - 1) + param_pp + 1) && "M=N+p+1 not satisfied");

  int num_seg = (knots_p.size() - 1) - 2 * param_pp;  // M-2*p

  // Stack the control points in matrices
  Eigen::Matrix<double, 2, -1> qp_matrix(2, qp.size());
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
  /// CONSTRUCT THE PIECE-WISE POLYNOMIAL FOR POSITION
  /////////////////////////////////////////////////////////////////////
  Eigen::Matrix<double, 4, 4> M;
  M << 1, 4, 1, 0,   //////
      -3, 0, 3, 0,   //////
      3, -6, 3, 0,   //////
      -1, 3, -3, 1;  //////
  M = M / 6.0;       // *1/3!

  pwp_p.clear();

  for (int i = param_pp; i < (param_pp + num_seg + 1); i++)  // i < knots.size() - p
  {
    pwp_p.times.push_back(knots_p(i));
  }

  for (int j = 0; j < num_seg; j++)
  {
    Eigen::Matrix<double, 4, 1> cps_x = (qp_matrix.block(0, j, 1, 4).transpose());
    Eigen::Matrix<double, 4, 1> cps_y = (qp_matrix.block(1, j, 1, 4).transpose());

    pwp_p.all_coeff_x.push_back((M * cps_x).reverse());  // at^3 + bt^2 + ct + d --> [a b c d]'
    pwp_p.all_coeff_y.push_back((M * cps_y).reverse());  // at^3 + bt^2 + ct + d --> [a b c d]'
  }
}

//
// ------------------------------------------------------------------------------------------------------
//

ConvexHullsOfCurves Panther::convexHullsOfCurves(double t_start, double t_end)
{
  ConvexHullsOfCurves result;

  for (auto traj : trajs_)
  {
    result.push_back(convexHullsOfCurve(traj, t_start, t_end));
  }

  return result;
}

//
// ------------------------------------------------------------------------------------------------------
//

ConvexHullsOfCurves Panther::convexHullsOfCurvesForObstacleEdge(double t_start, double t_end,
                                                                const Eigen::Affine3d& c_T_b,
                                                                const Eigen::Affine3d& w_T_b)
{
  ConvexHullsOfCurves result;

  mtx_trajs_.lock();
  for (auto traj : trajs_)
  {
    Eigen::Vector3d w_pos = evalMeanDynTrajCompiled(traj, t_start);

    //
    // Check if this trajectory is in FOV
    //

    if (par_.impose_FOV_in_trajCB)
    {
      Eigen::Vector3d c_pos = c_T_b * (w_T_b.inverse()) * w_pos;  // position of the obstacle in the camera frame
                                                                  // (i.e., depth optical frame)
      bool inFOV =                                                // check if it's inside the field of view.
          c_pos.z() < par_.fov_depth &&                           //////////////////////
          fabs(atan2(c_pos.x(), c_pos.z())) < ((par_.fov_x_deg * M_PI / 180.0) / 2.0) &&  ///// Note that fov_x_deg means x camera_depth_optical_frame
          fabs(atan2(c_pos.y(), c_pos.z())) < ((par_.fov_y_deg * M_PI / 180.0) / 2.0);  ///// Note that fov_y_deg means x camera_depth_optical_frame

      if (inFOV == false)
      {
        t_start = traj.time_received;
        t_end = t_start + par_.obstacle_visualization_duration;
      }
    }
    result.push_back(convexHullsOfCurve(traj, t_start, t_end));
  }
  mtx_trajs_.unlock();

  return result;
}

//
// ------------------------------------------------------------------------------------------------------
//

ConvexHullsOfCurve Panther::convexHullsOfCurve(mt::dynTrajCompiled& traj, double t_start, double t_end)
{
  ConvexHullsOfCurve convexHulls;
  double deltaT = (t_end - t_start) / (1.0 * par_.num_seg);  // num_seg is the number of intervals

  for (int i = 0; i <= par_.num_seg; i++)
  {
    convexHulls.push_back(convexHullOfInterval(traj, t_start + i * deltaT, t_start + (i + 1) * deltaT));
  }

  return convexHulls;
}

//
// ------------------------------------------------------------------------------------------------------
//

// See https://doc.cgal.org/Manual/3.7/examples/Convex_hull_3/quickhull_3.cpp
CGAL_Polyhedron_3 Panther::convexHullOfInterval(mt::dynTrajCompiled& traj, double t_start, double t_end)
{

  std::vector<Eigen::Vector3d> points = vertexesOfInterval(traj, t_start, t_end);

  std::vector<Point_3> points_cgal;
  for (auto point_i : points)
  {
    points_cgal.push_back(Point_3(point_i.x(), point_i.y(), point_i.z()));
  }

  if (points_cgal.size() == 0){
    std::cout << "points_cgal.size(): " << points_cgal.size() << std::endl;
    return CGAL_Polyhedron_3();
  } else {
    return cu::convexHullOfPoints(points_cgal);
  }

}