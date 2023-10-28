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

class TermGoalSender:

    def __init__(self):

        # mode
        self.mode = rospy.get_param('mode', 0) #default value is 0

        # circle radius
        self.circle_radius = rospy.get_param('circle_radius', 5.0) #default value is 5.0

        # home yet?
        self.is_home = False

        # position change
        self.sign = 1

        # initialization done?
        self.is_init_pos = False

        # reached goal?
        self.is_arrived = False

        # term_goal init
        self.term_goal=PoseStamped()
        self.term_goal.header.frame_id='world'
        self.pubTermGoal = rospy.Publisher('term_goal', PoseStamped, queue_size=1, latch=True)
        
        # state_pos init ()
        self.state_pos=np.array([0.0, 0.0, 0.0])

        # every 0.01 sec timerCB is called back
        self.timer = rospy.Timer(rospy.Duration(0.01), self.timerCB)

        # send goal
        self.sendGoal()

        # set initial time and how long the demo is
        self.time_init = rospy.get_rostime()
        self.total_secs = 1000.0; # sec

    def timerCB(self, event):

        # term_goal in array form
        self.term_goal_pos=np.array([self.term_goal.pose.position.x,self.term_goal.pose.position.y,self.term_goal.pose.position.z])

        # distance
        dist=np.linalg.norm(self.term_goal_pos-self.state_pos)

        # check distance and if it's close enough publish new term_goal
        dist_limit = 0.5
        if (dist < dist_limit):
            self.sendGoal()

    def sendGoal(self):

        # set terminal goals 
        angle = math.pi/2 * self.mode + math.pi/4
        self.term_goal.pose.position.x = - self.sign * self.circle_radius*math.cos(angle)
        self.term_goal.pose.position.y = - self.sign * self.circle_radius*math.sin(angle)
        self.term_goal.pose.position.z = 3.0
        self.is_arrived = not self.is_arrived
        self.sign = self.sign * (-1)
        self.pubTermGoal.publish(self.term_goal)

        return

    def stateCB(self, data):
        if not self.is_init_pos:
            self.init_pos = np.array([data.pos.x, data.pos.y, data.pos.z])
            self.is_init_pos = True

        self.state_pos = np.array([data.pos.x, data.pos.y, data.pos.z])

def startNode():
    c = TermGoalSender()
    rospy.Subscriber("state", State, c.stateCB)
    rospy.spin()

if __name__ == '__main__':
    rospy.init_node('TermGoalSender')
    startNode()