#!/usr/bin/env python

# /* ----------------------------------------------------------------------------
#  * Copyright 2023, Kota Kondo, Aerospace Controls Laboratory
#  * Massachusetts Institute of Technology
#  * All Rights Reserved
#  * Authors: Kota Kondo, et al.
#  * See LICENSE file for the license information
#  * -------------------------------------------------------------------------- */

import roslib
import rospy
import math
from snapstack_msgs.msg import Goal, State
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
import numpy as np
from numpy import linalg as LA
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_about_axis, quaternion_multiply, random_quaternion
from visualization_msgs.msg import Marker
import tf

class DroneMarkerPublisher:

    def __init__(self):

        self.world=PoseStamped()
        self.world.pose.position.x = rospy.get_param('~x', 0.0)
        self.world.pose.position.y = rospy.get_param('~y', 0.0)
        self.world.pose.position.z = rospy.get_param('~z', 0.0)
        yaw = rospy.get_param('~yaw', 0.0); pitch=0.0; roll=0.0
        quat = quaternion_from_euler(yaw, pitch, roll, 'rzyx')
        self.world.pose.orientation.x = quat[0]
        self.world.pose.orientation.y = quat[1]
        self.world.pose.orientation.z = quat[2]
        self.world.pose.orientation.w = quat[3]

        name = rospy.get_namespace()
        self.name = name[1:-1]

        self.pubMarkerDrone = rospy.Publisher('drone_marker', Marker, queue_size=1, latch=True)
        self.timer = rospy.Timer(rospy.Duration(0.01), self.publish_drone_marker)
        rospy.sleep(1.0)

    def worldCB(self, msg):
        self.world.pose.position.x = msg.pose.position.x
        self.world.pose.position.y = msg.pose.position.y
        self.world.pose.position.z = msg.pose.position.z
        self.world.pose.orientation.x = msg.pose.orientation.x
        self.world.pose.orientation.y = msg.pose.orientation.y
        self.world.pose.orientation.z = msg.pose.orientation.z
        self.world.pose.orientation.w = msg.pose.orientation.w

    def publish_drone_marker(self, timer):
        marker=Marker()
        marker.id=1
        marker.ns="mesh_"+self.name
        marker.header.frame_id="world"
        marker.type=marker.MESH_RESOURCE
        marker.action=marker.ADD
        marker.pose.position.x=self.world.pose.position.x
        marker.pose.position.y=self.world.pose.position.y
        marker.pose.position.z=self.world.pose.position.z
        marker.pose.orientation.x=self.world.pose.orientation.x
        marker.pose.orientation.y=self.world.pose.orientation.y
        marker.pose.orientation.z=self.world.pose.orientation.z
        marker.pose.orientation.w=self.world.pose.orientation.w
        marker.lifetime = rospy.Duration.from_sec(0.0)
        marker.mesh_use_embedded_materials=True
        marker.mesh_resource="package://panther_gazebo/meshes/quadrotor/quadrotor.dae"
        marker.scale.x=0.75
        marker.scale.y=0.75
        marker.scale.z=0.75
        self.pubMarkerDrone.publish(marker)  

def startNode():
    c = DroneMarkerPublisher()
    rospy.Subscriber("world", PoseStamped, c.worldCB)
    rospy.spin()

if __name__ == '__main__':

    ns = rospy.get_namespace()
    try:
        rospy.init_node('drone_marker')
        if str(ns) == '/':
            rospy.logfatal("Need to specify namespace as vehicle name.")
            rospy.logfatal("This is tyipcally accomplished in a launch file.")
            rospy.logfatal("Command line: ROS_NAMESPACE=mQ01 $ rosrun quad_control joy.py")
        else:
            print ("Starting drone_marker_publisher node for: " + ns)
            startNode()
    except rospy.ROSInterruptException:
        pass
