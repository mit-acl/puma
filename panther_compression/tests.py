from compression.utils.other import ActionManager, posAccelYaw2TfMatrix, State, ObservationManager
import numpy as np

np.set_printoptions(linewidth=np.nan)

am=ActionManager();

action=am.getRandomAction()
print(action.shape)

# p0=np.zeros((3,1));
# v0=np.zeros((3,1));
# a0=np.zeros((3,1));

# y0=np.zeros((1,));
# y_dot0=np.zeros((1,));


p0=np.random.rand(3,1);
v0=np.random.rand(3,1);
a0=np.random.rand(3,1);

y0=np.random.rand(1,1);
y_dot0=np.random.rand(1,1);

print("p0= \n", p0)
print("v0= \n", v0)
print("a0= \n", a0)

print("y0= \n", y0)
print("y_dot0= \n", y_dot0)

print("action= \n", action)

my_state=State(p0, v0, a0, y0, y_dot0)

w_posBS, w_yawBS= am.action2wBS(action, my_state)



# print("\npos_ctrl_pts=\n", pos_ctrl_pts)
# print("\nyaw_ctrl_pts=\n", yaw_ctrl_pts)
# print("\ntotal_time=\n", total_time)

# print(action.	)


print("================================")

# posAccelYaw2TransfMatrix(p0, a0, y0)


om=ObservationManager();

obs=om.getRandomObservation();
print("Random Observation")
print(obs)
print("f_v =", om.getf_v(obs))
print("f_a =", om.getf_a(obs))
print("getyaw_dot =", om.getyaw_dot(obs))
print("getf_g =", om.getf_g(obs))
print("getObstacles =", om.getObstacles(obs))