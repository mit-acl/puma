/* ----------------------------------------------------------------------------
 * Copyright 2021, Jesus Tordesillas Torres, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Jesus Tordesillas, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */
#include "utils.hpp"

#include "termcolor.hpp"

panther_msgs::Log log2LogMsg(mt::log log)
{
  panther_msgs::Log log_msg;

  log_msg.replanning_was_needed = log.replanning_was_needed;

  log_msg.ms_initial_setup = log.tim_initial_setup.getMsSaved();
  log_msg.ms_convex_hulls = log.tim_convex_hulls.getMsSaved();
  //log_msg.ms_opt = log.tim_opt.getMsSaved();
  log_msg.ms_opt = log.time_opt;
  log_msg.ms_guess_pos = log.tim_guess_pos.getMsSaved();
  log_msg.ms_guess_yaw_search_graph = log.tim_guess_yaw_search_graph.getMsSaved();
  log_msg.ms_guess_yaw_fit_poly = log.tim_guess_yaw_fit_poly.getMsSaved();
  log_msg.ms_total_replan = log.tim_total_replan.getMsSaved();

  log_msg.cost = log.cost;
  log_msg.obst_avoidance_violation = log.obst_avoidance_violation;
  log_msg.dyn_lim_violation = log.dyn_lim_violation;

  log_msg.tracking_now_pos.x = log.tracking_now_pos.x();
  log_msg.tracking_now_pos.y = log.tracking_now_pos.y();
  log_msg.tracking_now_pos.z = log.tracking_now_pos.z();

  log_msg.tracking_now_vel.x = log.tracking_now_vel.x();
  log_msg.tracking_now_vel.y = log.tracking_now_vel.y();
  log_msg.tracking_now_vel.z = log.tracking_now_vel.z();

  log_msg.pos.x = log.pos.x();
  log_msg.pos.y = log.pos.y();
  log_msg.pos.z = log.pos.z();

  log_msg.G_term_pos.x = log.G_term_pos.x();
  log_msg.G_term_pos.y = log.G_term_pos.y();
  log_msg.G_term_pos.z = log.G_term_pos.z();

  log_msg.success_guess_pos = log.success_guess_pos;
  log_msg.success_guess_yaw = log.success_guess_yaw;
  log_msg.success_opt = log.success_opt;
  log_msg.success_replanning = log.success_replanning;

  log_msg.info_replan = log.info_replan;
  log_msg.header.stamp = ros::Time::now();

  switch (log.drone_status)
  {
    case DroneStatus::TRAVELING:
      log_msg.drone_status = "TRAVELING";
      break;
    case DroneStatus::GOAL_SEEN:
      log_msg.drone_status = "GOAL_SEEN";
      break;
    case DroneStatus::GOAL_REACHED:
      log_msg.drone_status = "GOAL_REACHED";
      break;
    default:
      log_msg.drone_status = "";
      break;
  }

  return log_msg;
}

// https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function
double cdfUnivariateNormalDist(double x, double mu, double std_deviation)
{
  return 0.5 * (1 + erf((x - mu) / (std_deviation * sqrt(2))));
}

double getMinTimeDoubleIntegrator1D(const double& p0, const double& v0, const double& pf, const double& vf,
                                    const double& v_max, const double& a_max)
{
  // See also minimum_time.m
  // %The notation of this function is based on the paper "Constrained time-optimal
  // %control of double integrator system and its application in MPC"
  // % https://iopscience.iop.org/article/10.1088/1742-6596/783/1/012024

  double x1 = v0;
  double x2 = p0;
  double x1r = vf;
  double x2r = pf;

  double k1 = a_max;  // Note that the paper uses u\in[-1, 1].But setting k1 to a_max has the same effect k2 = 1.0;
  double k2 = 1.0;

  double x1_bar = v_max;

  double B = (k2 / (2 * k1)) * sign(-x1 + x1r) * (pow(x1, 2) - pow(x1r, 2)) + x2r;
  double C = (k2 / (2 * k1)) * (pow(x1, 2) + pow(x1r, 2)) - (k2 / k1) * pow(x1_bar, 2) + x2r;
  double D = (-k2 / (2 * k1)) * (pow(x1, 2) + pow(x1r, 2)) + (k2 / k1) * pow(x1_bar, 2) + x2r;

  double time;

  if ((x2 <= B) && (x2 >= C))
  {
    time = (-k2 * (x1 + x1r) +
            2 * sqrt(pow(k2, 2) * pow(x1, 2) - k1 * k2 * ((k2 / (2 * k1)) * (pow(x1, 2) - pow(x1r, 2)) + x2 - x2r))) /
           (k1 * k2);
  }

  else if ((x2 <= B) && (x2 < C))
  {
    time = (x1_bar - x1 - x1r) / k1 + (pow(x1, 2) + pow(x1r, 2)) / (2 * k1 * x1_bar) + (x2r - x2) / (k2 * x1_bar);
  }

  else if ((x2 > B) && (x2 <= D))
  {
    time = (k2 * (x1 + x1r) +
            2 * sqrt(pow(k2, 2) * pow(x1, 2) + k1 * k2 * ((k2 / (2 * k1)) * (-pow(x1, 2) + pow(x1r, 2)) + x2 - x2r))) /
           (k1 * k2);
  }

  else
  {  // (x2 > B) && (x2 > D)

    time = (x1_bar + x1 + x1r) / k1 + (pow(x1, 2) + pow(x1r, 2)) / (2 * k1 * x1_bar) + (-x2r + x2) / (k2 * x1_bar);
  }

  return time;
}

double getMinTimeDoubleIntegrator3D(const Eigen::Vector3d& p0, const Eigen::Vector3d& v0, const Eigen::Vector3d& pf,
                                    const Eigen::Vector3d& vf, const Eigen::Vector3d& v_max,
                                    const Eigen::Vector3d& a_max)
{
  double min_x = getMinTimeDoubleIntegrator1D(p0.x(), v0.x(), pf.x(), vf.x(), v_max.x(), a_max.x());
  double min_y = getMinTimeDoubleIntegrator1D(p0.y(), v0.y(), pf.y(), vf.y(), v_max.y(), a_max.y());
  double min_z = getMinTimeDoubleIntegrator1D(p0.z(), v0.z(), pf.z(), vf.z(), v_max.z(), a_max.z());

  double min_time = std::max({ min_x, min_y, min_z });  // Note that it's the maximum of all the axes

  return min_time;
}

double getMinTimeDoubleIntegrator3DFromState(mt::state initial_state, mt::state final_state,
                                             const Eigen::Vector3d& v_max, const Eigen::Vector3d& a_max)
{
  return getMinTimeDoubleIntegrator3D(initial_state.pos, initial_state.vel, final_state.pos, final_state.vel, v_max,
                                      a_max);
}

// note that b>=a is a requirement
double probUnivariateNormalDistAB(double a, double b, double mu, double std_deviation)
{
  if (b < a)
  {
    /////// Debugging
    std::cout << "Needed: b>=a. ABORTING!" << std::endl;
    abort();
    ////////////////
  }

  return (cdfUnivariateNormalDist(b, mu, std_deviation) - cdfUnivariateNormalDist(a, mu, std_deviation));
}

// This works when the covariance matris is diagonal (and the diagonal is given by the vector std_deviation.^2 )
// TODO: we are assumming that all the vectors have the same size (if not it'll crash) --> make template
double probMultivariateNormalDist(const Eigen::VectorXd& a, const Eigen::VectorXd& b, const Eigen::VectorXd& mu,
                                  const Eigen::VectorXd& std_deviation)
{
  double prob_less_a = 1.0;
  for (int i = 0; i < a.size(); i++)
  {
    prob_less_a *= cdfUnivariateNormalDist(a(i), mu(i), std_deviation(i));
  }

  double prob_less_b = 1.0;
  for (int i = 0; i < b.size(); i++)
  {
    prob_less_b *= cdfUnivariateNormalDist(b(i), mu(i), std_deviation(i));
  }
  return (prob_less_b - prob_less_a);
}

visualization_msgs::MarkerArray pwp2ColoredMarkerArray(mt::PieceWisePol& pwp, double t_init, double t_final,
                                                       int samples, std::string ns, Eigen::Vector3d& color)
{
  visualization_msgs::MarkerArray marker_array;

  if (t_final < t_init)
  {
    // std::cout << "t_final<t_init" << std::endl;
    abort();
    return marker_array;
  }

  // std::cout << "t_init= " << std::setprecision(15) << t_init << std::endl;
  // std::cout << "t_final= " << std::setprecision(15) << t_final << std::endl;

  double deltaT = (t_final - t_init) / (1.0 * samples);

  geometry_msgs::Point p_last = eigen2point(pwp.eval(t_init));

  int j = 7 * 9000;  // TODO

  for (double t = t_init; t <= t_final; t = t + deltaT)
  {
    visualization_msgs::Marker m;
    m.type = visualization_msgs::Marker::ARROW;
    m.header.frame_id = "map";
    m.header.stamp = ros::Time::now();
    m.ns = ns;
    m.action = visualization_msgs::Marker::ADD;
    m.id = j;

    m.color.r = color.x();  // color(RED_NORMAL);
    m.color.g = color.y();
    m.color.b = color.z();
    m.color.a = 1.0;

    m.scale.x = 0.1;
    m.scale.y = 0.0000001;  // rviz complains if not
    m.scale.z = 0.0000001;  // rviz complains if not

    m.pose.orientation.w = 1.0;

    geometry_msgs::Point p = eigen2point(pwp.eval(t));

    m.points.push_back(p_last);
    m.points.push_back(p);

    p_last = p;
    marker_array.markers.push_back(m);
    j = j + 1;
  }

  return marker_array;
}

// https://www.geeksforgeeks.org/binomial-coefficient-dp-9/#highlighter_72035:~:text=Following%20is%20a%20space%2Doptimized%20version%20of%20the%20above%20code
int nChoosek(int n, int k)
{
  int C[k + 1];
  memset(C, 0, sizeof(C));

  C[0] = 1;  // nC0 is 1

  for (int i = 1; i <= n; i++)
  {
    // Compute next row of pascal triangle using
    // the previous row
    for (int j = std::min(i, k); j > 0; j--)
      C[j] = C[j] + C[j - 1];
  }
  return C[k];
}

// https://stackoverflow.com/questions/141422/how-can-a-transform-a-polynomial-to-another-coordinate-system

// Denote T=[t^n,...,t,1]'
// given a polynomial p(t)=coeff_p'*T
// compute the coeff_q of the polynomial q(t)=p(a*t + b).
// q(t)=coeff_q'*T
void linearTransformPoly(const Eigen::VectorXd& coeff_p, Eigen::VectorXd& coeff_q, double a, double b)
{
  Eigen::VectorXd c = coeff_p.reverse();

  Eigen::MatrixXd Q(coeff_p.size(), coeff_p.size());
  for (int i = 0; i < Q.rows(); i++)
  {
    for (int j = 0; j < Q.cols(); j++)
    {
      int jchoosei = (i > j) ? 0 : nChoosek(j, i);
      Q(i, j) = jchoosei * pow(a, i) * pow(b, j - i);
    }
  }
  Eigen::VectorXd d = Q * c;
  coeff_q = d.reverse();  // Note that d = d.reverse() is wrong!

  // std::cout << "Q= \n" << Q << std::endl;
  // std::cout << "coeff_p= \n" << coeff_p << std::endl;
  // std::cout << "coeff_q= \n" << coeff_q << std::endl;
  // std::cout << "coeff_q= \n" << coeff_q << std::endl;
}

// Obtain the coefficients of q(t) such that
// p(tp1)==q(tq1)
// p(tp2)==q(tq2)
void changeDomPoly(const Eigen::VectorXd& coeff_p, double tp1, double tp2, Eigen::VectorXd& coeff_q, double tq1,
                   double tq2)
{
  // The equations below are from executing this in Matlab:
  // syms tq1 tq2 tp1 tp2 a b;
  // s=solve([tp1==a*tq1+b, tp2==a*tq2+b],[a,b])
  double a = (tp1 - tp2) / (tq1 - tq2);
  double b = -(tp1 * tq2 - tp2 * tq1) / (tq1 - tq2);
  linearTransformPoly(coeff_p, coeff_q, a, b);
}

panther_msgs::PieceWisePolTraj pwp2PwpMsg(const mt::PieceWisePol& pwp)
{
  panther_msgs::PieceWisePolTraj pwp_msg;

  for (int i = 0; i < pwp.times.size(); i++)
  {
    // std::cout << termcolor::red << "in pwp2PwpMsg, pushing back" << std::setprecision(20) << pwp.times[i]
    //           << termcolor::reset << std::endl;
    pwp_msg.times.push_back(pwp.times[i]);
  }

  // push x
  for (auto coeff_x_i : pwp.all_coeff_x)
  {
    panther_msgs::CoeffPoly coeff_poly3;

    for (int i = 0; i < coeff_x_i.size(); i++)
    {
      coeff_poly3.data.push_back(coeff_x_i(i));
    }
    pwp_msg.all_coeff_x.push_back(coeff_poly3);
  }

  // push y
  for (auto coeff_y_i : pwp.all_coeff_y)
  {
    panther_msgs::CoeffPoly coeff_poly3;
    for (int i = 0; i < coeff_y_i.size(); i++)
    {
      coeff_poly3.data.push_back(coeff_y_i(i));
    }
    pwp_msg.all_coeff_y.push_back(coeff_poly3);
  }

  // push z
  for (auto coeff_z_i : pwp.all_coeff_z)
  {
    panther_msgs::CoeffPoly coeff_poly3;
    for (int i = 0; i < coeff_z_i.size(); i++)
    {
      coeff_poly3.data.push_back(coeff_z_i(i));
    }
    pwp_msg.all_coeff_z.push_back(coeff_poly3);
  }

  return pwp_msg;
}

void verify(bool cond, std::string info_if_false)
{
  if (cond == false)
  {
    std::cout << termcolor::bold << termcolor::red << info_if_false << termcolor::reset << std::endl;
    std::cout << termcolor::red << "Aborting" << termcolor::reset << std::endl;
    abort();
  }
}

mt::PieceWisePol pwpMsg2Pwp(const panther_msgs::PieceWisePolTraj& pwp_msg)
{
  mt::PieceWisePol pwp;

  if (pwp_msg.all_coeff_x.size() != pwp_msg.all_coeff_y.size() ||
      pwp_msg.all_coeff_x.size() != pwp_msg.all_coeff_z.size())
  {
    std::cout << " coeff_x,coeff_y,coeff_z of pwp_msg should have the same elements" << std::endl;
    std::cout << " ================================" << std::endl;
    abort();
  }

  for (int i = 0; i < pwp_msg.times.size(); i++)
  {
    pwp.times.push_back(pwp_msg.times[i]);
  }

  for (int i = 0; i < pwp_msg.all_coeff_x.size(); i++)  // For each of the intervals
  {
    int degree = pwp_msg.all_coeff_x[i].data.size() - 1;  // Should be the same for all i, and for x, y, z

    Eigen::VectorXd tmp_x(degree + 1);
    Eigen::VectorXd tmp_y(degree + 1);
    Eigen::VectorXd tmp_z(degree + 1);

    for (int j = 0; j < (degree + 1); j++)
    {
      tmp_x(j) = pwp_msg.all_coeff_x[i].data[j];
      tmp_y(j) = pwp_msg.all_coeff_y[i].data[j];
      tmp_z(j) = pwp_msg.all_coeff_z[i].data[j];
    }

    // std::cout << termcolor::on_blue << "pwpMsg2Pwp: " << pwp_msg.all_coeff_z[i].a << ", " << pwp_msg.all_coeff_z[i].b
    // << ", "
    //           << pwp_msg.all_coeff_z[i].c << ", " << pwp_msg.all_coeff_z[i].d << termcolor::reset << std::endl;

    pwp.all_coeff_x.push_back(tmp_x);
    pwp.all_coeff_y.push_back(tmp_y);
    pwp.all_coeff_z.push_back(tmp_z);
  }

  return pwp;
}

visualization_msgs::Marker edges2Marker(const mt::Edges& edges, std_msgs::ColorRGBA color_marker)
{
  visualization_msgs::Marker marker;

  if (edges.size() == 0)  // there are no edges
  {
    // std::cout << "there are no edges" << std::endl;
    return marker;
  }

  marker.header.frame_id = "map";
  marker.header.stamp = ros::Time::now();
  marker.ns = "markertacles";
  marker.id = 0;
  marker.type = marker.LINE_LIST;
  marker.action = marker.ADD;
  marker.pose = identityGeometryMsgsPose();

  marker.points.clear();

  for (auto edge : edges)
  {
    marker.points.push_back(eigen2point(edge.first));
    marker.points.push_back(eigen2point(edge.second));
  }

  marker.scale.x = 0.03;
  // marker.scale.y = 0.00001;
  // marker.scale.z = 0.00001;
  marker.color = color_marker;

  return marker;
}

mt::PieceWisePol createPwpFromStaticPosition(const mt::state& current_state)
{
  mt::PieceWisePol pwp;
  pwp.times = { ros::Time::now().toSec(), ros::Time::now().toSec() + 1e10 };

  // In this case we encode it as a third-order polynomial
  Eigen::Matrix<double, 4, 1> coeff_x_interv0;  // [a b c d]' of the interval 0
  Eigen::Matrix<double, 4, 1> coeff_y_interv0;  // [a b c d]' of the interval 0
  Eigen::Matrix<double, 4, 1> coeff_z_interv0;  // [a b c d]' of the interval 0

  coeff_x_interv0 << 0.0, 0.0, 0.0, current_state.pos.x();
  coeff_y_interv0 << 0.0, 0.0, 0.0, current_state.pos.y();
  coeff_z_interv0 << 0.0, 0.0, 0.0, current_state.pos.z();

  pwp.all_coeff_x.push_back(coeff_x_interv0);
  pwp.all_coeff_y.push_back(coeff_y_interv0);
  pwp.all_coeff_z.push_back(coeff_z_interv0);

  return pwp;
}

// returns a mt::PieceWisePol, taking the polynomials of p1 and p2 that should satisfy p1>t, p2>t
mt::PieceWisePol composePieceWisePol(const double t, const double dc, mt::PieceWisePol& p1, mt::PieceWisePol& p2)
{
  // if t is in between p1 and p2, force p2[0] to be t
  if (t > p1.times.back() && t < p2.times.front())  // && fabs(t - p2.times.front()) <= dc) TODO Sometimes fabs(t -
                                                    // p2.times.front()) is 0.18>c
  {
    p2.times.front() = t;
  }

  if (p1.times.back() < p2.times.front())  // && fabs(p1.times.back() - p2.times.front()) <= dc) TODO
  {
    p2.times.front() = p1.times.back();
  }

  if (t < p1.times.front())  // TODO
  {
    p1.times.front() = t;
  }

  if (fabs(t - p2.times.front()) < 1e-5)
  {
    return p2;
  }

  if (p1.times.back() < p2.times.front() || t > p2.times.back() || t < p1.times.front())
  {
    // TODO?
    // std::cout << "Error composing the piecewisePol" << std::endl;
    // std::cout << std::setprecision(30) << "t= " << t << std::endl;
    // std::cout << std::setprecision(30) << "p1.times.front()= " << p1.times.front() << std::endl;
    // std::cout << std::setprecision(30) << "p1.times.back()= " << p1.times.back() << std::endl;
    // std::cout << std::setprecision(30) << "p2.times.front() = " << p2.times.front() << std::endl;
    // std::cout << std::setprecision(30) << "p2.times.back() = " << p2.times.back() << std::endl;
    mt::PieceWisePol tmp;
    return tmp;
  }

  std::vector<int> indexes1, indexes2;

  for (int i = 0; i < p1.times.size(); i++)
  {
    if (p1.times[i] > t && p1.times[i] < p2.times[0])
    {
      indexes1.push_back(i);
    }
  }

  for (int i = 0; i < p2.times.size(); i++)
  {
    if (p2.times[i] > t)
    {
      indexes2.push_back(i);
    }
  }

  mt::PieceWisePol p;
  p.times.push_back(t);

  for (auto index_1_i : indexes1)
  {
    p.times.push_back(p1.times[index_1_i]);
    p.all_coeff_x.push_back(p1.all_coeff_x[index_1_i - 1]);
    p.all_coeff_y.push_back(p1.all_coeff_y[index_1_i - 1]);
    p.all_coeff_z.push_back(p1.all_coeff_z[index_1_i - 1]);
  }

  for (auto index_2_i : indexes2)
  {
    if (index_2_i == 0)
    {
      p.all_coeff_x.push_back(p1.all_coeff_x.back());
      p.all_coeff_y.push_back(p1.all_coeff_y.back());
      p.all_coeff_z.push_back(p1.all_coeff_z.back());
      p.times.push_back(p2.times[index_2_i]);
      continue;
    }
    p.times.push_back(p2.times[index_2_i]);
    p.all_coeff_x.push_back(p2.all_coeff_x[index_2_i - 1]);
    p.all_coeff_y.push_back(p2.all_coeff_y[index_2_i - 1]);
    p.all_coeff_z.push_back(p2.all_coeff_z[index_2_i - 1]);
  }

  return p;
}

std::vector<std::string> pieceWisePol2String(const mt::PieceWisePol& pwp)
{
  // Define strings
  std::string s_x = "0.0";
  std::string s_y = "0.0";
  std::string s_z = "0.0";

  int deg = pwp.getDeg();

  //(pwp.times - 1) is the number of intervals
  for (int i = 0; i < (pwp.times.size() - 1); i++)  // i is the index of the interval
  {
    std::string div_by_delta = "/ (" + std::to_string(pwp.times[i + 1] - pwp.times[i]) + ")";

    std::string t = "(min(t," + std::to_string(pwp.times.back()) + "))";

    std::string u = "(" + t + "-" + std::to_string(pwp.times[i]) + ")" + div_by_delta;
    u = "(" + u + ")";

    // std::string u = "(" + t + "-" + std::to_string(pwp.times[i]) + ")" + div_by_delta;
    // u = "(" + u + ")";
    // std::string uu = u + "*" + u;
    // std::string uuu = u + "*" + u + "*" + u;
    /*    std::cout << "pwp.times[i]= " << pwp.times[i] << std::endl;
        std::cout << "pwp.times[i+1]= " << pwp.times[i + 1] << std::endl;*/

    std::string cond;
    if (i == (pwp.times.size() - 2))  // if the last interval
    {
      cond = "(t>=" + std::to_string(pwp.times[i]) + ")";
    }
    else
    {
      cond = "(t>=" + std::to_string(pwp.times[i]) + " and " + "t<" + std::to_string(pwp.times[i + 1]) + ")";
    }

    std::string s_x_i = "";
    std::string s_y_i = "";
    std::string s_z_i = "";
    for (int j = 0; j <= deg; j++)
    {
      std::string power_u = "(" + u + "^" + std::to_string(deg - j) + ")";

      s_x_i = s_x_i + "+" + std::to_string((double)pwp.all_coeff_x[i](j)) + "*" + power_u;
      s_y_i = s_y_i + "+" + std::to_string((double)pwp.all_coeff_y[i](j)) + "*" + power_u;
      s_z_i = s_z_i + "+" + std::to_string((double)pwp.all_coeff_z[i](j)) + "*" + power_u;
    }

    s_x_i = cond + "*(" + s_x_i + ")";
    s_y_i = cond + "*(" + s_y_i + ")";
    s_z_i = cond + "*(" + s_z_i + ")";

    // std::string s_x_i = std::to_string((double)pwp.all_coeff_x[i](0)) + "*" + uuu;   //////////////////
    // s_x_i = s_x_i + "+" + std::to_string((double)pwp.all_coeff_x[i](1)) + "*" + uu;  //////////////////
    // s_x_i = s_x_i + "+" + std::to_string((double)pwp.all_coeff_x[i](2)) + "*" + u;   //////////////////
    // s_x_i = s_x_i + "+" + std::to_string((double)pwp.all_coeff_x[i](3));             //////////////////
    // s_x_i = cond + "*(" + s_x_i + ")";

    // std::string s_y_i = std::to_string((double)pwp.all_coeff_y[i](0)) + "*" + uuu;   //////////////////
    // s_y_i = s_y_i + "+" + std::to_string((double)pwp.all_coeff_y[i](1)) + "*" + uu;  //////////////////
    // s_y_i = s_y_i + "+" + std::to_string((double)pwp.all_coeff_y[i](2)) + "*" + u;   //////////////////
    // s_y_i = s_y_i + "+" + std::to_string((double)pwp.all_coeff_y[i](3));             //////////////////
    // s_y_i = cond + "*(" + s_y_i + ")";

    // std::string s_z_i = std::to_string((double)pwp.all_coeff_z[i](0)) + "*" + uuu;   //////////////////
    // s_z_i = s_z_i + "+" + std::to_string((double)pwp.all_coeff_z[i](1)) + "*" + uu;  //////////////////
    // s_z_i = s_z_i + "+" + std::to_string((double)pwp.all_coeff_z[i](2)) + "*" + u;   //////////////////
    // s_z_i = s_z_i + "+" + std::to_string((double)pwp.all_coeff_z[i](3));             //////////////////
    // s_z_i = cond + "*(" + s_z_i + ")";

    s_x = s_x + " + " + s_x_i;
    s_y = s_y + " + " + s_y_i;
    s_z = s_z + " + " + s_z_i;
  }

  std::vector<std::string> s;
  s.push_back(s_x);
  s.push_back(s_y);
  s.push_back(s_z);

  return s;
}

void printStateDeque(std::deque<mt::state>& data)
{
  for (int i = 0; i < data.size(); i++)
  {
    data[i].printHorizontal();
  }
}

void printStateVector(std::vector<mt::state>& data)
{
  for (int i = 0; i < data.size(); i++)
  {
    data[i].printHorizontal();
  }
}

geometry_msgs::Pose identityGeometryMsgsPose()
{
  geometry_msgs::Pose pose;
  pose.position.x = 0;
  pose.position.y = 0;
  pose.position.z = 0;
  pose.orientation.x = 0;
  pose.orientation.y = 0;
  pose.orientation.z = 0;
  pose.orientation.w = 1;
  return pose;
}

std_msgs::ColorRGBA getColorInterpBetween2Colors(double v, double vmin, double vmax, std_msgs::ColorRGBA min_color,
                                                 std_msgs::ColorRGBA max_color)
{
  double dv;

  if (v < vmin)
    v = vmin;
  if (v > vmax)
    v = vmax;
  dv = vmax - vmin;

  double fraction_max_color = (v - vmin) / dv;

  // std_msgs::ColorRGBA first = color(YELLOW_NORMAL);
  // std_msgs::ColorRGBA last = color(GREEN_NORMAL);

  std_msgs::ColorRGBA c;
  c.r = (max_color.r - min_color.r) * fraction_max_color + min_color.r;
  c.g = (max_color.g - min_color.g) * fraction_max_color + min_color.g;
  c.b = (max_color.b - min_color.b) * fraction_max_color + min_color.b;
  c.a = 1;

  return c;
}

std_msgs::ColorRGBA getColorJet(double v, double vmin, double vmax)
{
  vmax = std::max(vmax, 1.0001 * vmin);  // Ensure vmax>vmin

  std_msgs::ColorRGBA c;
  c.r = 1;
  c.g = 1;
  c.b = 1;
  c.a = 1;
  // white
  double dv;

  if (v < vmin)
    v = vmin;
  if (v > vmax)
    v = vmax;
  dv = vmax - vmin;

  if (v < (vmin + 0.25 * dv))
  {
    c.r = 0;
    c.g = 4 * (v - vmin) / dv;
  }
  else if (v < (vmin + 0.5 * dv))
  {
    c.r = 0;
    c.b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
  }
  else if (v < (vmin + 0.75 * dv))
  {
    c.r = 4 * (v - vmin - 0.5 * dv) / dv;
    c.b = 0;
  }
  else
  {
    c.g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
    c.b = 0;
  }

  return (c);
}

std_msgs::ColorRGBA color(int id)
{
  std_msgs::ColorRGBA red;
  red.r = 1;
  red.g = 0;
  red.b = 0;
  red.a = 1;
  std_msgs::ColorRGBA red_trans;
  red_trans.r = 1;
  red_trans.g = 0;
  red_trans.b = 0;
  red_trans.a = 0.7;
  std_msgs::ColorRGBA red_trans_trans;
  red_trans_trans.r = 1;
  red_trans_trans.g = 0;
  red_trans_trans.b = 0;
  red_trans_trans.a = 0.4;
  std_msgs::ColorRGBA blue;
  blue.r = 0;
  blue.g = 0;
  blue.b = 1;
  blue.a = 1;
  std_msgs::ColorRGBA blue_trans;
  blue_trans.r = 0;
  blue_trans.g = 0;
  blue_trans.b = 1;
  blue_trans.a = 0.7;
  std_msgs::ColorRGBA blue_trans_trans;
  blue_trans_trans.r = 0;
  blue_trans_trans.g = 0;
  blue_trans_trans.b = 1;
  blue_trans_trans.a = 0.4;
  std_msgs::ColorRGBA blue_light;
  blue_light.r = 0.5;
  blue_light.g = 0.7;
  blue_light.b = 1;
  blue_light.a = 1;
  std_msgs::ColorRGBA green;
  green.r = 0;
  green.g = 1;
  green.b = 0;
  green.a = 1;
  std_msgs::ColorRGBA yellow;
  yellow.r = 1;
  yellow.g = 1;
  yellow.b = 0;
  yellow.a = 1;
  std_msgs::ColorRGBA orange_trans;  // orange transparent
  orange_trans.r = 1;
  orange_trans.g = 0.5;
  orange_trans.b = 0;
  orange_trans.a = 0.7;
  std_msgs::ColorRGBA teal_normal;  // teal transparent
  teal_normal.r = 25 / 255.0;
  teal_normal.g = 1.0;
  teal_normal.b = 240.0 / 255.0;
  teal_normal.a = 1.0;
  std_msgs::ColorRGBA black_trans;  // black transparent
  black_trans.r = 0.0;
  black_trans.g = 0.0;
  black_trans.b = 0.0;
  black_trans.a = 0.2;
  std_msgs::ColorRGBA black;  // black
  black.r = 0.0;
  black.g = 0.0;
  black.b = 0.0;
  black.a = 1.0;

  switch (id)
  {
    case RED_NORMAL:
      return red;
      break;
    case RED_TRANS:
      return red_trans;
      break;
    case RED_TRANS_TRANS:
      return red_trans_trans;
      break;
    case BLUE_NORMAL:
      return blue;
      break;
    case BLUE_TRANS:
      return blue_trans;
      break;
    case BLUE_TRANS_TRANS:
      return blue_trans_trans;
      break;
    case BLUE_LIGHT:
      return blue_light;
      break;
    case GREEN_NORMAL:
      return green;
      break;
    case YELLOW_NORMAL:
      return yellow;
      break;
    case ORANGE_TRANS:
      return orange_trans;
      break;
    case BLACK_TRANS:
      return black_trans;
      break;
    case BLACK_NORMAL:
      return black;
      break;
    case TEAL_NORMAL:
      return teal_normal;
      break;
    default:
      std::cout << "COLOR NOT DEFINED, returning RED" << std::endl;
      return red;
  }
}

// It assummes that the box is aligned with x, y, z
// C1 is the corner with lowest x,y,z
// C2 is the corner with highest x,y,z
// center is the center of the sphere
// r is the radius of the sphere
bool boxIntersectsSphere(Eigen::Vector3d center, double r, Eigen::Vector3d c1, Eigen::Vector3d c2)
{
  // https://developer.mozilla.org/en-US/docs/Games/Techniques/3D_collision_detection
  //(Section Sphere vs AABB)

  Eigen::Vector3d closest_point;  // closest point from the center of the sphere to the box

  closest_point(0) = std::max(c1.x(), std::min(center.x(), c2.x()));
  closest_point(1) = std::max(c1.y(), std::min(center.y(), c2.y()));
  closest_point(2) = std::max(c1.z(), std::min(center.z(), c2.z()));

  // this is the same as isPointInsideSphere
  double distance_to_closest_point = (center - closest_point).norm();

  return (distance_to_closest_point < r);  // true if the box intersects the sphere
}

//## From Wikipedia - http://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
void quaternion2Euler(tf2::Quaternion q, double& roll, double& pitch, double& yaw)
{
  tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
}

void quaternion2Euler(Eigen::Quaterniond q, double& roll, double& pitch, double& yaw)
{
  tf2::Quaternion tf_q(q.x(), q.y(), q.z(), q.w());
  quaternion2Euler(tf_q, roll, pitch, yaw);
}

void quaternion2Euler(geometry_msgs::Quaternion q, double& roll, double& pitch, double& yaw)
{
  tf2::Quaternion tf_q(q.x, q.y, q.z, q.w);
  quaternion2Euler(tf_q, roll, pitch, yaw);
}

void saturate(Eigen::Vector3d& tmp, const Eigen::Vector3d& min, const Eigen::Vector3d& max)
{
  saturate(tmp(0), min(0), max(0));
  saturate(tmp(1), min(1), max(1));
  saturate(tmp(2), min(2), max(2));
}

void saturate(int& var, const int min, const int max)
{
  if (var < min)
  {
    var = min;
  }
  else if (var > max)
  {
    var = max;
  }
}

// TODO: Make a template here
void saturate(double& var, const double min, const double max)
{
  if (var < min)
  {
    var = min;
  }
  else if (var > max)
  {
    var = max;
  }
}

visualization_msgs::Marker getMarkerSphere(double scale, int my_color)
{
  visualization_msgs::Marker marker;

  marker.header.frame_id = "map";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::SPHERE;
  marker.scale.x = scale;
  marker.scale.y = scale;
  marker.scale.z = scale;
  marker.color = color(my_color);

  return marker;
}

double angleBetVectors(const Eigen::Vector3d& a, const Eigen::Vector3d& b)
{
  double tmp = a.dot(b) / (a.norm() * b.norm());

  saturate(tmp, -1, 1);
  return acos(tmp);
}

void angle_wrap(double& diff)
{
  diff = fmod(diff + M_PI, 2 * M_PI);
  if (diff < 0)
    diff += 2 * M_PI;
  diff -= M_PI;
}

geometry_msgs::Point pointOrigin()
{
  geometry_msgs::Point tmp;
  tmp.x = 0;
  tmp.y = 0;
  tmp.z = 0;
  return tmp;
}

Eigen::Vector3d vec2eigen(geometry_msgs::Vector3 vector)
{
  Eigen::Vector3d tmp;
  tmp << vector.x, vector.y, vector.z;
  return tmp;
}

geometry_msgs::Vector3 eigen2rosvector(Eigen::Vector3d vector)
{
  geometry_msgs::Vector3 tmp;
  tmp.x = vector(0, 0);
  tmp.y = vector(1, 0);
  tmp.z = vector(2, 0);
  return tmp;
}

std::vector<double> eigen2std(const Eigen::Vector3d& v)
{
  return std::vector<double>{ v.x(), v.y(), v.z() };
}

// std::vector<float> eigen2std(const Eigen::Vector3f& v)
// {
//   return std::vector<float>{ v.x(), v.y(), v.z() };
// }

geometry_msgs::Point eigen2point(Eigen::Vector3d vector)
{
  geometry_msgs::Point tmp;
  tmp.x = vector[0];
  tmp.y = vector[1];
  tmp.z = vector[2];
  return tmp;
}

geometry_msgs::Vector3 vectorNull()
{
  geometry_msgs::Vector3 tmp;
  tmp.x = 0;
  tmp.y = 0;
  tmp.z = 0;
  return tmp;
}

geometry_msgs::Vector3 vectorUniform(double a)
{
  geometry_msgs::Vector3 tmp;
  tmp.x = a;
  tmp.y = a;
  tmp.z = a;
  return tmp;
}

Eigen::Matrix<double, 9, 1> buildVarianceVector(double sigma_pos_x, double sigma_vel_x, double sigma_acc_x,
                                                double sigma_pos_y, double sigma_vel_y, double sigma_acc_y,
                                                double sigma_pos_z, double sigma_vel_z, double sigma_acc_z)
{
  Eigen::Matrix<double, 9, 1> tmp;
  tmp << sigma_pos_x, sigma_vel_x, sigma_acc_x, sigma_pos_y, sigma_vel_y, sigma_acc_y, sigma_pos_z, sigma_vel_z,
      sigma_acc_z;
  return tmp;
}

Eigen::Matrix<double, 9, 1> buildVarianceVector(Eigen::Vector3d sigma_pos, Eigen::Vector3d sigma_vel,
                                                Eigen::Vector3d sigma_acc)
{
  Eigen::Matrix<double, 9, 1> tmp;
  tmp << sigma_pos(0), sigma_vel(0), sigma_acc(0), sigma_pos(1), sigma_vel(1), sigma_acc(1), sigma_pos(2), sigma_vel(2),
      sigma_acc(2);
  return tmp;
}

visualization_msgs::MarkerArray trajectory2ColoredMarkerArray(const mt::trajectory& data, double max_value, int increm,
                                                              std::string ns, double scale, std::string color_type,
                                                              int id_agent, int n_agents, double min_aug_cost,
                                                              double max_aug_cost, double aug_cost, bool collides)
{
  visualization_msgs::MarkerArray marker_array;

  if (data.size() == 0)
  {
    return marker_array;
  }
  geometry_msgs::Point p_last;
  p_last.x = data[0].pos(0);
  p_last.y = data[0].pos(1);
  p_last.z = data[0].pos(2);

  increm = (increm < 1.0) ? 1 : increm;

  int j = 9000;
  for (int i = 0; i < data.size(); i = i + increm)
  {
    double vel = data[i].vel.norm();
    visualization_msgs::Marker m;
    m.type = visualization_msgs::Marker::ARROW;
    m.header.frame_id = "map";
    m.header.stamp = ros::Time::now();
    m.ns = ns;
    m.action = visualization_msgs::Marker::ADD;
    m.id = j;
    if (color_type == "vel")  // TODO: "vel" is hand-coded
    {
      m.color = getColorJet(vel, 0, max_value);  // note that par_.v_max is per axis!
    }
    else if (color_type == "time")  // TODO: "time" is hand-coded
    {
      m.color = getColorJet(i, 0, data.size());
    }
    else if (color_type == "aug_cost")  // TODO: "time" is hand-coded
    {
      m.color = getColorJet(aug_cost, min_aug_cost, max_aug_cost);
    }
    else if (color_type == "agent")  // TODO: "time" is hand-coded
    {
      m.color = getColorJet(id_agent, 0, n_agents);
    }
    else if (color_type == "black")  // TODO: "time" is hand-coded
    {
      m.color = color(BLACK_NORMAL);
    }
    else
    {
      std::cout << "color_type CHOSEN IS NOT SUPPORTED" << std::endl;
      abort();
    }

    if (collides)
    {
      m.color.a = 0.0;  // Make it invisible
    }

    // m.color.a = alpha;
    m.scale.x = scale;
    m.scale.y = 0.0000001;  // rviz complains if not
    m.scale.z = 0.0000001;  // rviz complains if not

    m.pose.orientation.w = 1.0;
    // std::cout << "Mandando bloque" << X.block(i, 0, 1, 3) << std::endl;
    geometry_msgs::Point p;
    p.x = data[i].pos(0);
    p.y = data[i].pos(1);
    p.z = data[i].pos(2);
    m.points.push_back(p_last);
    m.points.push_back(p);
    // std::cout << "pushing marker\n" << m << std::endl;
    p_last = p;
    marker_array.markers.push_back(m);
    j = j + 1;
  }
  return marker_array;
}
