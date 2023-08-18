#!/usr/bin/env python
# subscribe to T265's raw images and distort it and run it on FastSAM, and get the mean of the blobs
# Author: Kota Kondo
# Atribution: Jouko Kinnari (https://gitlab.com/mit-acl/clipper/uncertain-localization/depth-estimation-experiments/)

import os
import argparse
from geometry_msgs.msg import PoseStamped
from panther_msgs.msg import Drift
import rospy
import numpy as np
from utils import get_quaternion_from_euler, quaternion_to_euler_angle_vectorized1
import tf
from scipy.spatial.transform import Rotation as Rot
from motlee.utils.transform import transform

class POSE_CORRUPTER_ROS:

    def __init__(self):

        ##
        ## get parameters
        ##

        # get namespace
        self.ns = rospy.get_namespace()
        
        # constant drift
        self.is_constant_drift = rospy.get_param('~is_constant_drift', False)
        self.constant_drift_x = rospy.get_param('~constant_drift_x', 0.0)
        self.constant_drift_y = rospy.get_param('~constant_drift_y', 0.0)
        self.constant_drift_z = rospy.get_param('~constant_drift_z', 0.0)
        self.constant_drift_roll = rospy.get_param('~constant_drift_roll', 0.0)
        self.constant_drift_pitch = rospy.get_param('~constant_drift_pitch', 0.0)
        self.constant_drift_yaw = rospy.get_param('~constant_drift_yaw', 0.0)
        self.constant_drift_quaternion = get_quaternion_from_euler(self.constant_drift_roll, self.constant_drift_pitch, self.constant_drift_yaw)

        # linear drift
        self.is_linear_drift = rospy.get_param('~is_linear_drift', False)
        self.linear_drift_rate_x = rospy.get_param('~linear_drift_rate_x', 0.0)
        self.linear_drift_rate_y = rospy.get_param('~linear_drift_rate_y', 0.0)
        self.linear_drift_rate_z = rospy.get_param('~linear_drift_rate_z', 0.0)
        self.linear_drift_rate_roll = rospy.get_param('~linear_drift_rate_roll', 0.0)
        self.linear_drift_rate_pitch = rospy.get_param('~linear_drift_rate_pitch', 0.0)
        self.linear_drift_rate_yaw = rospy.get_param('~linear_drift_rate_yaw', 0.0)

        self.is_initial_time_set = False

        self.initial_time_buffer = 0.0 # seconds

        ##
        ## set up ROS communications
        ##

        rospy.Subscriber('world', PoseStamped, self.corrupt_and_publish_pose)
        self.pub_world = rospy.Publisher('corrupted_world', PoseStamped, queue_size=1)
        self.pub_drift = rospy.Publisher('drift', Drift, queue_size=1)

    # no change publish
    def pub_pose_msg_without_drift(self, pose_msg, pub_pose_msg):
        pub_pose_msg.pose.position.x = pose_msg.pose.position.x
        pub_pose_msg.pose.position.y = pose_msg.pose.position.y
        pub_pose_msg.pose.position.z = pose_msg.pose.position.z
        pub_pose_msg.pose.orientation.x = pose_msg.pose.orientation.x
        pub_pose_msg.pose.orientation.y = pose_msg.pose.orientation.y
        pub_pose_msg.pose.orientation.z = pose_msg.pose.orientation.z
        pub_pose_msg.pose.orientation.w = pose_msg.pose.orientation.w
        return pub_pose_msg
    
    # publish pose
    def corrupt_and_publish_pose(self, pose_msg):

        # the the initial time for linear drift
        if self.is_initial_time_set == False:
            self.initial_time = rospy.Time.now()
            self.is_initial_time_set = True

        pub_pose_msg = PoseStamped()
        pub_pose_msg.header.stamp = rospy.Time.now()
        pub_pose_msg.header.frame_id = "world"

        drift_pos = [0.0, 0.0, 0.0]
        drift_euler = [0.0, 0.0, 0.0]

        # if it's still before the inital time buffer, don't add drift
        if (rospy.Time.now() - self.initial_time).to_sec() < self.initial_time_buffer:
            pub_pose_msg = self.pub_pose_msg_without_drift(pose_msg, pub_pose_msg)
        elif self.is_constant_drift: # add constant drift

            # if you introduce pose drift, you need to change the position in terms of the drift
            T = np.eye(4)
            T[:3,3] = [self.constant_drift_x, self.constant_drift_y, self.constant_drift_z]
            T[:3,:3] = Rot.from_euler('xyz', [self.constant_drift_roll, self.constant_drift_pitch, self.constant_drift_yaw]).as_matrix()
            rotated_position = transform(T, np.array([pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z]))

            pub_pose_msg.pose.position.x = rotated_position[0] 
            pub_pose_msg.pose.position.y = rotated_position[1]
            pub_pose_msg.pose.position.z = rotated_position[2]

            corrupted_quaternion, drift_euler = self.corrupt_quaternion_constant_drift(pose_msg)
            drift_pos = [self.constant_drift_x, self.constant_drift_y, self.constant_drift_z]
            pub_pose_msg.pose.orientation.x = corrupted_quaternion.x
            pub_pose_msg.pose.orientation.y = corrupted_quaternion.y
            pub_pose_msg.pose.orientation.z = corrupted_quaternion.z
            pub_pose_msg.pose.orientation.w = corrupted_quaternion.w

        elif self.is_linear_drift: # add linear drift
            pub_pose_msg.pose.position.x = pose_msg.pose.position.x + self.linear_drift_rate_x * round(((rospy.Time.now() - self.initial_time).to_sec() - self.initial_time_buffer),2)
            pub_pose_msg.pose.position.y = pose_msg.pose.position.y + self.linear_drift_rate_y * round(((rospy.Time.now() - self.initial_time).to_sec() - self.initial_time_buffer),2)
            pub_pose_msg.pose.position.z = pose_msg.pose.position.z + self.linear_drift_rate_z * round(((rospy.Time.now() - self.initial_time).to_sec() - self.initial_time_buffer),2)
            corrupted_quaternion, drift_euler = self.corrupt_quaternion_linear_drift(pose_msg)
            drift_pos = [self.linear_drift_rate_x * round(((rospy.Time.now() - self.initial_time).to_sec() - self.initial_time_buffer),2), \
                         self.linear_drift_rate_y * round(((rospy.Time.now() - self.initial_time).to_sec() - self.initial_time_buffer),2), \
                         self.linear_drift_rate_z * round(((rospy.Time.now() - self.initial_time).to_sec() - self.initial_time_buffer),2)]
            pub_pose_msg.pose.orientation.x = corrupted_quaternion.x
            pub_pose_msg.pose.orientation.y = corrupted_quaternion.y
            pub_pose_msg.pose.orientation.z = corrupted_quaternion.z
            pub_pose_msg.pose.orientation.w = corrupted_quaternion.w
        else: # no drift
            pub_pose_msg = self.pub_pose_msg_without_drift(pose_msg, pub_pose_msg)

        # publish pose as corrupted_world
        self.pub_world.publish(pub_pose_msg)

        # publish to tf
        br = tf.TransformBroadcaster()
        br.sendTransform((pub_pose_msg.pose.position.x, pub_pose_msg.pose.position.y, pub_pose_msg.pose.position.z),\
                            (pub_pose_msg.pose.orientation.x, pub_pose_msg.pose.orientation.y, pub_pose_msg.pose.orientation.z, pub_pose_msg.pose.orientation.w),\
                            rospy.Time.now(),\
                            self.ns + "corrupted_world",\
                            "world")
        
        # publish drift
        drift_msg = Drift()
        drift_msg.header.stamp = rospy.Time.now()
        drift_msg.header.frame_id = "world"
        drift_msg.drift_pos = drift_pos
        drift_msg.drift_euler = drift_euler
        self.pub_drift.publish(drift_msg)
    
    # corrupt quaternion for constant drift
    def corrupt_quaternion_constant_drift(self, pose_msg):
        euler = quaternion_to_euler_angle_vectorized1(pose_msg.pose.orientation.w, pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z)
        roll = euler[0] + self.constant_drift_roll
        pitch = euler[1] + self.constant_drift_pitch
        yaw = euler[2] + self.constant_drift_yaw
        quaternion = get_quaternion_from_euler(roll, pitch, yaw)
        drift_euler = [self.constant_drift_roll, self.constant_drift_pitch, self.constant_drift_yaw]
        return quaternion, drift_euler


    # corrupt quaternion for linear drift
    def corrupt_quaternion_linear_drift(self, pose_msg):
        euler = quaternion_to_euler_angle_vectorized1(pose_msg.pose.orientation.w, pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z)
        roll = euler[0] + self.linear_drift_rate_roll * ((rospy.Time.now() - self.initial_time).to_sec() - self.initial_time_buffer)
        pitch = euler[1] + self.linear_drift_rate_pitch * ((rospy.Time.now() - self.initial_time).to_sec() - self.initial_time_buffer)
        yaw = euler[2] + self.linear_drift_rate_yaw * ((rospy.Time.now() - self.initial_time).to_sec() - self.initial_time_buffer)
        quaternion = get_quaternion_from_euler(roll, pitch, yaw)
        drift_euler = [self.linear_drift_rate_roll * ((rospy.Time.now() - self.initial_time).to_sec() - self.initial_time_buffer), \
                            self.linear_drift_rate_pitch * ((rospy.Time.now() - self.initial_time).to_sec() - self.initial_time_buffer), \
                            self.linear_drift_rate_yaw * ((rospy.Time.now() - self.initial_time).to_sec() - self.initial_time_buffer)]
        return quaternion, drift_euler
if __name__ == '__main__':
    rospy.init_node('pose_corrupter')
    pose_corrupter = POSE_CORRUPTER_ROS()
    rospy.spin()