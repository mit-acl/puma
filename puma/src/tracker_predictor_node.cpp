/* ----------------------------------------------------------------------------
 * Copyright 2023, Kota Kondo, Aerospace Controls Laboratory
 * Massachusetts Institute of Technology
 * All Rights Reserved
 * Authors: Jesus Tordesillas, et al.
 * See LICENSE file for the license information
 * -------------------------------------------------------------------------- */

#include "tracker_predictor.hpp"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "cluster_node");
  ros::NodeHandle nh("~");
  TrackerPredictor tmp(nh);
  ros::spin();
}
