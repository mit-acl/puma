#!/usr/bin/env python

# /* ----------------------------------------------------------------------------
#  * Copyright 2021, Jesus Tordesillas Torres, Aerospace Controls Laboratory
#  * Massachusetts Institute of Technology
#  * All Rights Reserved
#  * Authors: Jesus Tordesillas, et al.
#  * See LICENSE file for the license information
#  * -------------------------------------------------------------------------- */

import rospy
from panther_msgs.msg import WhoPlans
from snapstack_msgs.msg import Goal, State
from geometry_msgs.msg import Pose, PoseStamped
from snapstack_msgs.msg import QuadFlightMode
import math
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import sys
import numpy as np
from sympy import Piecewise
from math import sin, cos, tan #To make sure eval('sin(5)') works
from visualization_msgs.msg import Marker

import time
import string
import re

class ObstaclePlanner:

    def __init__(self):
 
        tmp=rospy.get_namespace();
        #https://stackoverflow.com/questions/1450897/remove-characters-except-digits-from-string-using-python
        tmp=re.sub("\D", "", tmp)

        # Who plans
        self.whoplans=WhoPlans()
        self.whoplans.value=self.whoplans.OTHER

        # get params from yaml
        self.traj_x = parse_expr(self.safeGetParam('~traj_x'))
        self.traj_y = parse_expr(self.safeGetParam('~traj_y'))
        self.traj_z = parse_expr(self.safeGetParam('~traj_z'))
        
        # initialization
        self.state_initialized = False
        self.position=[0.0, 0.0, 0.0]
        self.traj=np.array([self.traj_x, self.traj_y, self.traj_z])

        # publisher
        self.pubGoal = rospy.Publisher('goal', Goal, queue_size=1)
        self.pubGoalTimer=rospy.Timer(rospy.Duration(0.01), self.pubCB)
        self.pubGoalTimer.shutdown()
        self.initialize_planner()

    def safeGetParam(self, param):
        result=""
        if(rospy.has_param(param)):
            return rospy.get_param(param)
        else:
            print("____________________________________________________")
            print("Param ", rospy.resolve_name(param), " does NOT exist")
            print("____________________________________________________")
            sys.exit()

    #In rospy, the callbacks are all of them in separate threads
    def stateCB(self, data):
        self.position=[data.pos.x, data.pos.y, data.pos.z]

        if(self.state_initialized==False):
            self.state_initialized=True

    def diffSimpyArray(self, myarray):
        t=sp.symbols('t');
        return np.array([sp.diff(myarray[i],t) for i in range(len(myarray))])       

    def evalSimpyArray(self, tt, myarray):
        t=sp.symbols('t');
        return np.array([float(myarray[i].subs(t,tt)) for i in range(len(myarray))])

    def evalTraj(self,tt):
        return self.evalSimpyArray(tt, self.traj)

    # def evalTrajDot(self,tt):
    #     return self.evalSimpyArray(tt, self.traj_dot)


    #Fit a third degree polynomial passing through the points (t0,xt0), and (t1,xt1), and with specified derivatives at those points
    #Result taken running this with Matlab:
    #     syms t xt0 vt0 xt1 vt1 t0 t1 a b c d real
    #     f=a*t^3+b*t^2+c*t+d;
    #     s=solve([subs(f,t,t0)==xt0, subs(f,t,t1)==xt1, subs(diff(f),t,t0)==vt0,  subs(diff(f),t,t1)==vt1],[a b c d]);
    #     [s.a s.b s.c s.d]'
    #Then remember to substitute ^ by **
    def fit3rdDegPol(self, t0_original, t1_original, xt0, xt1, vt0, vt1):

        #shifting to avoid numerical issues. Note that no scaling is done, and thereore the velocities are valid
        t0=t0_original-t0_original; #i.e. 0.0
        t1=t1_original-t0_original; 

        a=-(2*xt0 - 2*xt1 - t0*vt0 - t0*vt1 + t1*vt0 + t1*vt1)/(t0 - t1)**3
        b=(3*t0*xt0 - 3*t0*xt1 + 3*t1*xt0 - 3*t1*xt1 - t0**2*vt0 - 2*t0**2*vt1 + 2*t1**2*vt0 + t1**2*vt1 - t0*t1*vt0 + t0*t1*vt1)/(t0 - t1)**3
        c=(t0**3*vt1 - t1**3*vt0 - t0*t1**2*vt0 + 2*t0**2*t1*vt0 - 2*t0*t1**2*vt1 + t0**2*t1*vt1 - 6*t0*t1*xt0 + 6*t0*t1*xt1)/(t0 - t1)**3
        d=(t0**3*xt1 - t1**3*xt0 + t0*t1**3*vt0 - t0**3*t1*vt1 + 3*t0*t1**2*xt0 - 3*t0**2*t1*xt1 - t0**2*t1**2*vt0 + t0**2*t1**2*vt1)/((t0 - t1)*(t0**2 - 2*t0*t1 + t1**2))
        t_shifted=sp.symbols('t_shifted');
        t=sp.symbols('t');
        return (a*t_shifted**3+b*t_shifted**2+c*t_shifted+d).subs(t_shifted,t-t0_original);

    def initialize_planner(self):

        upper_bound_time_s = 2.0; #This simply takes into account the time spent on this function (so tha)

        t_init_function=rospy.get_time();

        #x0 -->  position at t=t_current + upper_bound_time_s
        #x1 = traj(t1)
        #x2 = traj(t2)
        #v2 = (dtraj(t)/dt) for t=t2
        #t0-->t1 [LINE] Straight line (with constant velocity v) that connects x0 --> x1
        #t1-->t2 [POLY] Third degree polynomial f(t) such that f(t1)==x1, f'(t1)==v, f(t2)==x2, f'(t2)==v2 [This is to ensure a smooth transition]
        #t>t2 [TRAJ] Follow traj
        t0=rospy.get_time()+upper_bound_time_s; #Same as before, but it's float
        t=sp.symbols('t'); #ros time

        delta01=7.0;
        delta12=3.0;

        t1=t0+delta01;
        t2=t1+delta12;
        
        t1=2.2;
        t0=t1-delta01;
        t2=t1+delta12;

        x0=self.position;
        x1=self.evalTraj(t1);
        x2=self.evalTraj(t2);

        v=(x1-x0)/delta01
        my_line=x0 + v*(t-t0);

        v1=self.evalSimpyArray(t1,self.diffSimpyArray(my_line));
        v2=self.evalSimpyArray(t2,self.diffSimpyArray(self.traj));

        my_poly=np.array([self.fit3rdDegPol(t1,t2,x1[i],x2[i],v1[i],v2[i]) for i in range(len(self.traj))])

        # print(my_poly)

        cond0=(t-t0)<=delta01
        cond1=((t-t0)>delta01) & ((t-t0)<(delta01+delta12))
        cond2= True #rest of the cases   

        self.whole_traj =      np.array([ Piecewise( (my_line[i], cond0) ,                ( my_poly[i], cond1 ),                 ( self.traj[i], cond2 )  )  for i in range(len(self.traj))])
        # print(self.whole_traj)
        self.whole_traj_dot =  np.array([ Piecewise( (sp.diff(my_line[i],t), cond0) ,     ( sp.diff(my_poly[i],t), cond1 ) ,     ( sp.diff(self.traj[i],t), cond2 )   )  for i in range(len(self.traj))])
        self.whole_traj_dot2 = np.array([ Piecewise( (sp.diff(my_line[i],t,t), cond0) ,   ( sp.diff(my_poly[i],t,t), cond1 ) ,   ( sp.diff(self.traj[i],t,t), cond2 )   )  for i in range(len(self.traj))])
        self.whole_traj_dot3 = np.array([ Piecewise( (sp.diff(my_line[i],t,t,t), cond0) , ( sp.diff(my_poly[i],t,t,t), cond1 ) , ( sp.diff(self.traj[i],t,t,t), cond2 )   )  for i in range(len(self.traj))])

        total_duration_function= rospy.get_time()-t_init_function;
        rospy.sleep(upper_bound_time_s - total_duration_function)
        # self.pubGoalTimer.run()

        self.time_end_init=rospy.get_time();
        self.t0=t0;

        self.pubGoalTimer=rospy.Timer(rospy.Duration(0.01), self.pubCB)
        print("End of initialize_planner()")

    def abort_planner(self):
        self.pubGoalTimer.shutdown()

    def whoplansCB(self,data):
        if(self.state_initialized==False):
            while not rospy.is_shutdown():
                print("Waiting until state is initialized") #Note that stateCB runs in parallel!

        if(data.value==data.PANTHER and self.whoplans.value==self.whoplans.OTHER):
            self.initialize_planner()
            self.whoplans=data;
        elif(data.value==data.OTHER and self.whoplans.value==self.whoplans.PANTHER):
            self.abort_planner()
            self.whoplans=data;


    def pubCB(self, timer):
        goal=Goal()
        goal.header.stamp=rospy.Time.now();
        ti=self.t0+(rospy.get_time()-self.time_end_init); #Same as before, but it's float

        whole_traj_ti=self.evalSimpyArray(ti, self.whole_traj)
        whole_traj_dot_ti=self.evalSimpyArray(ti, self.whole_traj_dot)
        whole_traj_dot2_ti=self.evalSimpyArray(ti, self.whole_traj_dot2)
        whole_traj_dot3_ti=self.evalSimpyArray(ti, self.whole_traj_dot3)

        goal.p.x=whole_traj_ti[0]
        goal.p.y=whole_traj_ti[1]
        goal.p.z=whole_traj_ti[2]

        goal.v.x=whole_traj_dot_ti[0]
        goal.v.y=whole_traj_dot_ti[1]
        goal.v.z=whole_traj_dot_ti[2]

        goal.a.x=whole_traj_dot2_ti[0]
        goal.a.y=whole_traj_dot2_ti[1]
        goal.a.z=whole_traj_dot2_ti[2]

        goal.j.x=whole_traj_dot3_ti[0]
        goal.j.y=whole_traj_dot3_ti[1]
        goal.j.z=whole_traj_dot3_ti[2]

        goal.psi = 0.0
        goal.power= True; #Turn on the motors

        if(self.whoplans.value==self.whoplans.PANTHER):
            self.pubGoal.publish(goal)
                  
def startNode():
    c = ObstaclePlanner()
    rospy.Subscriber("state", State, c.stateCB)
    rospy.Subscriber("who_plans",WhoPlans,c.whoplansCB) 
    rospy.spin()

if __name__ == '__main__':
    rospy.init_node('obstacle_planner')  
    startNode()
    print("started!") 