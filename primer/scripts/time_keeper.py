#!/usr/bin/env python

import math
import os
import time
import rospy
import rosgraph
from geometry_msgs.msg import PoseStamped
from snapstack_msgs.msg import State
import numpy as np
from random import *
from rosgraph_msgs.msg import Clock
from panther_msgs.msg import GoalReached

class TimeKeeper:

    def __init__(self):
        
        # simulation time
        self.sim_time = rospy.get_param('~sim_time', 100.0) #default value is 100.0

        # time
        self.tic = 0.0
        self.toc = 0.0

        # publisher (it's not really agents reached goal, but to be consistent with goal_checker we name this topic this way)
        self.end_pub = rospy.Publisher('/sim_all_agents_goal_reached', GoalReached, queue_size=1, latch=True)

        # pub msg initialization
        self.end_msg = GoalReached()

        # counter
        self.cnt = 0

    def clock_cb(self, data):

        # get time
        self.toc = data.clock.secs + data.clock.nsecs*1e-9

        # print time
        if self.cnt % 100 == 0: 
            print("Time: ", self.toc - self.tic)
        self.cnt += 1

        # if time is up then sent shutdown signal
        if self.toc - self.tic > self.sim_time:
            print("time is up")
            self.end_msg.header.stamp = rospy.get_rostime()
            self.end_msg.is_goal_reached = True
            self.end_pub.publish(self.end_msg)

def startNode():
    c = TimeKeeper()
    rospy.Subscriber("/clock", Clock, c.clock_cb)
    rospy.spin()

if __name__ == '__main__':
    rospy.init_node('TimeKeeper')
    startNode()