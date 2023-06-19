/* ----------------------------------------------------------------------------
 * Copyright 2021, Jesus Tordesillas Torres, Aerospace Controls Laboratory
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

  std::cout << "About to setup callback\n";

  TrackerPredictor tmp(nh);

  // ros::Subscriber sub = nh.subscribe("cloud", 1, &TrackerPredictor::cloud_cb, &tmp);

  // pub_cluster0 = nh.advertise<sensor_msgs::PointCloud2>("cluster_0", 1);
  // objID_pub = nh.advertise<std_msgs::Int32MultiArray>("obj_id", 1);

  // markerPub = nh.advertise<visualization_msgs::MarkerArray>("viz", 1);

  // ros::Rate r(10); // 100 hz
  //   while (ros::ok()){
  //   	try {
  //   		ros::spinOnce();
  //   	}
  //   	catch(...){
  //   		std::cout<<"Exception!"<<std::endl;
  //   	}
  //   }

  //   r.sleep();

  ros::spin();
}
