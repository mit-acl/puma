/* ----------------------------------------------------------------------------
 * Copyright 2021, Jesus Tordesillas Torres, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Jesus Tordesillas, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

#pragma once

#include <Eigen/Dense>

#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <rviz_visual_tools/rviz_visual_tools.h>
#include <snapstack_msgs/State.h>
#include <snapstack_msgs/Goal.h>
#include <std_srvs/Empty.h>

#include <panther_msgs/WhoPlans.h>
#include <panther_msgs/DynTraj.h>

#include <mutex>
#include <map>
#include <string>

#include "utils.hpp"
#include "panther.hpp"
#include "panther_types.hpp"

#include "timer.hpp"

#include <costmap_converter/ObstacleArrayMsg.h>
#include <costmap_converter/ObstacleMsg.h>


// #define WHOLE 1  // Whole trajectory (part of which is planned on unkonwn space)
// #define SAFE 2   // Safe path

namespace rvt = rviz_visual_tools;

class PantherRos
{
public:
  PantherRos(ros::NodeHandle nh1, ros::NodeHandle nh2, ros::NodeHandle nh3, ros::NodeHandle nh4, ros::NodeHandle nh5);
  ~PantherRos();

private:
  std::unique_ptr<Panther> panther_ptr_;

  // publishing traj to other agents
  void publishOwnTraj(const mt::PieceWisePol& pwp, const bool& is_committed);
  void publishPlanes(std::vector<Hyperplane3D>& planes);

  // class methods
  void pubTraj(const std::vector<mt::state>& data);
  // void pubBestTrajs(const std::vector<si::solOrGuess>& best_solutions);

  visualization_msgs::MarkerArray pubVectorOfsolOrGuess(const std::vector<si::solOrGuess>& sols_or_guesses,
                                                        ros::Publisher& publisher, std::string ns,
                                                        std::string color_type);

  void terminalGoalCB(const geometry_msgs::PoseStamped& msg);
  void pubState(const mt::state& msg, const ros::Publisher pub);
  void stateCB(const snapstack_msgs::State& msg);
  void whoPlansCB(const panther_msgs::WhoPlans& msg);
  void pubCB(const ros::TimerEvent& e);
  void replanCB(const ros::TimerEvent& e);
  void obstacleEdgeCB(const ros::TimerEvent& e);
  void publishObstacleCB(const ros::TimerEvent& e);
  void publishOwnTrajInFailure(mt::Edges edges_obstacles);
  void trajCB(const panther_msgs::DynTraj& msg);
  void obstacleTrajCB(const panther_msgs::DynTraj& msg);
  void costmapObstCB(const costmap_converter::ObstacleArrayMsg& msg);

  void pauseTime();
  void unpauseTime();

  // void clearMarkerSetOfArrows();
  void clearMarkerActualTraj();
  void clearMarkerColoredTraj();
  void clearObstacleEdges();
  void clearObstacleUncertaintyEdges();
  void pubActualTraj();
  visualization_msgs::MarkerArray clearArrows();
  // geometry_msgs::Vector3 vectorNull();
  void clearMarkerArray(visualization_msgs::MarkerArray& tmp, ros::Publisher& publisher);
  void publishPoly(const vec_E<Polyhedron<3>>& poly);
  // visualization_msgs::MarkerArray Matrix2ColoredMarkerArray(Eigen::MatrixXd& X, int type);
  void publishText();
  void publishFOV();
  void pubObstacles(mt::Edges edges_obstacles);
  void pubObstaclesWithUncertainty(mt::Edges edges_obstacles_uncertainty);

  void pubUncertainty(const std::vector<Eigen::Vector3d>& obstacle_uncertainty_list,
                      const std::vector<Eigen::Matrix<double, 9, 1>>& obstacle_sigma_list,
                      const std::vector<double>& obstacle_uncertainty_times,
                      const std::vector<Eigen::Vector3d>& moving_direction_uncertainty_list,
                      const std::vector<Eigen::Matrix<double, 9, 1>>& moving_direction_sigma_list,
                      const std::vector<double>& moving_direction_uncertainty_times);

  void pubAlpha(double& alpha);
  std_msgs::Float64MultiArray vecEigen3dToFloat64MultiArray(const std::vector<Eigen::Vector3d>& vec_eigen);
  std_msgs::Float64MultiArray vecEigen9dToFloat64MultiArray(const std::vector<Eigen::Matrix<double, 9, 1>>& vec_eigen);
  std_msgs::Float64MultiArray vecDoubleToFloat64MultiArray(const std::vector<double>& vec_double);

  void frameAlignCB(const motlee_msgs::SE3Transform& msg);
  void getFrameAlignmentT(Eigen::Matrix4d& T, const double qw, const double qx, const double qy, const double qz, const double px, const double py, const double pz);
  void applyFrameAlignment(Eigen::Matrix4d& T, const std::string& frame_src);
  void getPosFromT(std::vector<double>& pos, const Eigen::Matrix4d& T);
  void constructFOVMarker();

  // dictory of frame alignment transform matrix
  std::map<std::string, Eigen::Matrix4d> dict_frame_align_;
  std::mutex mtx_frame_align_;

  mt::state state_;

  std::string world_name_ = "world";

  rvt::RvizVisualToolsPtr visual_tools_;

  visualization_msgs::Marker E_;
  visualization_msgs::Marker A_;
  visualization_msgs::Marker setpoint_;

  ros::NodeHandle nh1_;
  ros::NodeHandle nh2_;
  ros::NodeHandle nh3_;
  ros::NodeHandle nh4_;
  ros::NodeHandle nh5_;

  ros::Publisher pub_point_G_;
  ros::Publisher pub_point_G_term_;
  ros::Publisher pub_goal_;
  ros::Publisher pub_traj_safe_;
  ros::Publisher pub_setpoint_;
  ros::Publisher pub_actual_traj_;

  ros::Publisher pub_point_A_;

  ros::Publisher pub_traj_safe_colored_;

  ros::Publisher pub_best_solutions_expert_;
  ros::Publisher pub_best_solution_expert_;

  ros::Publisher pub_best_solutions_student_;
  ros::Publisher pub_best_solution_student_;

  ros::Publisher pub_obstacle_uncertainty_;
  ros::Publisher pub_obstacle_uncertainty_values_;
  ros::Publisher pub_obstacle_sigma_values_;
  ros::Publisher pub_obstacle_uncertainty_times_;
  ros::Publisher pub_moving_direction_uncertainty_values_;
  ros::Publisher pub_moving_direction_sigma_values_;
  ros::Publisher pub_moving_direction_uncertainty_times_;
  ros::Publisher pub_alpha_;

  ros::Publisher pub_guesses_;
  ros::Publisher pub_splines_fitted_;

  visualization_msgs::MarkerArray ma_best_solution_student_;
  visualization_msgs::MarkerArray ma_best_solutions_student_;
  visualization_msgs::MarkerArray ma_best_solution_expert_;
  visualization_msgs::MarkerArray ma_best_solutions_expert_;
  visualization_msgs::MarkerArray ma_guesses_;

  ros::Publisher pub_text_;
  ros::Publisher pub_traj_;

  ros::Publisher poly_safe_pub_;

  ros::Publisher pub_fov_;
  ros::Publisher pub_obstacles_;
  ros::Publisher pub_log_;
  ros::Publisher pub_is_ready_;
  ros::Publisher pub_pause_sim_;

  ros::Subscriber sub_term_goal_;
  ros::Subscriber sub_whoplans_;
  ros::Subscriber sub_state_;
  ros::Subscriber sub_frame_align_;
  ros::Subscriber sub_traj_;                    // subscriber for obs perfect traj prediction
  std::vector<ros::Subscriber> sub_traj_list_;  // subscribers for each agent
  ros::Subscriber sub_costmap_obst_;           // subscriber for costmap obstacles

  ros::Timer pubCBTimer_;
  ros::Timer replanCBTimer_;
  ros::Timer obstacleEdgeCBTimer_;
  ros::Timer obstacleShareCBTimer_;

  ros::ServiceClient pauseGazebo_;
  ros::ServiceClient unpauseGazebo_;
  std_srvs::Empty emptySrv_;

  mt::parameters par_;  // where all the parameters are

  std::string name_drone_;

  std::vector<std::string> traj_;  // trajectory of the dynamic obstacle

  visualization_msgs::MarkerArray traj_safe_colored_;

  int actual_trajID_ = 0;

  int id_;  // id of the drone

  bool published_initial_position_ = false;

  Eigen::Affine3d w_T_b_;
  Eigen::Affine3d c_T_b_;
  std::mutex mtx_w_T_b_;

  mt::PieceWisePol pwp_last_;

  PANTHER_timers::Timer timer_stop_;

  visualization_msgs::Marker marker_fov_;

  std::string name_camera_depth_optical_frame_tf_;

  panther_msgs::DynTraj obstacle_traj_;
  std::mutex mtx_obstacle_traj_;

  // store costmap obstacles
  std::vector<costmap_converter::ObstacleMsg> costmap_obst_;

};
