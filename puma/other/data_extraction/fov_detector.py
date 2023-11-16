
#!/usr/bin/env python

import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting
import math, time, random
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as Rot

def puma_specific_rotation(p0, delta_y, delta_z, fov_depth, agent_pos, agent_quat):

    # Define FOV vertices
    p1 = np.array([-delta_y, delta_z, fov_depth])
    p2 = np.array([delta_y, delta_z, fov_depth])
    p3 = np.array([delta_y, -delta_z, fov_depth])
    p4 = np.array([-delta_y, -delta_z, fov_depth])

    # rotatoin camera_depth_optical_frame to camera_link (refer to static_transform.launch)
    R2 = Rot.from_euler('xyz', [-1.57, 0, -1.57]) #TODO: hard-coded, and need to match static_transforms.launch
    p1 = R2.apply(p1)
    p2 = R2.apply(p2)
    p3 = R2.apply(p3)
    p4 = R2.apply(p4)

    # rotate points by agent_quat
    R1 = Rot.from_quat(agent_quat)
    p1 = R1.apply(p1)
    p2 = R1.apply(p2)
    p3 = R1.apply(p3)
    p4 = R1.apply(p4)

    # Another translation camera_link to body
    cameralink2body = np.array([0.1, 0, 0])

    # translate back the FOV vertices
    p0 = p0 + cameralink2body 
    p1 = p1 + np.array(agent_pos) + cameralink2body 
    p2 = p2 + np.array(agent_pos) + cameralink2body
    p3 = p3 + np.array(agent_pos) + cameralink2body
    p4 = p4 + np.array(agent_pos) + cameralink2body

    return p0, p1, p2, p3, p4

def check_obst_is_in_FOV(agent_pos, agent_quat, obst_pos, fov_x_deg, fov_y_deg, fov_depth):
    """
    Check if the obstacle is in the agent's FOV.
    """

    # Define v0 and obst_pos
    p0 = np.array(agent_pos)
    obst_pos = np.array(obst_pos)
    
    # Compute the FOV vertices wrt the origin (from panther_ros.cpp)
    delta_y = fov_depth * abs(math.tan((fov_x_deg * math.pi / 180) / 2.0))
    delta_z = fov_depth * abs(math.tan((fov_y_deg * math.pi / 180) / 2.0))

    p0, p1, p2, p3, p4 = puma_specific_rotation(p0, delta_y, delta_z, fov_depth, agent_pos, agent_quat)

    # Check if the obstacle is in the FOV
    poly = np.array([p0, p1, p2, p3, p4])
    point = obst_pos
    return Delaunay(poly).find_simplex(point) >= 0  # True if point lies within poly

def rotate_vector(v, q):
    """
    Rotate a vector v by a quaternion q.
    """
    q_conj = np.array([-q[0], -q[1], -q[2], q[3]])
    v_prime = quaternion_multiply(quaternion_multiply(q, np.append(v, 0)), q_conj)
    return v_prime[:3]

def quaternion_multiply(q0, q1):
    """
    Multiply two quaternions.
    """
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1

    Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    return np.array([Q0Q1_x, Q0Q1_y, Q0Q1_z, Q0Q1_w])

def visualization(agent_pos, agent_quat, obst_pos, fov_x_deg, fov_y_deg, fov_depth, t=0):
    # visualize the agent's FOV
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('FOV visualization')
    ax.set_box_aspect([1,1,1])
    ax.scatter(agent_pos[0], agent_pos[1], agent_pos[2], c='r', marker='o')
    
    # Compute the FOV vertices (from panther_ros.cpp)
    delta_y = fov_depth * abs(math.tan((fov_x_deg * math.pi / 180) / 2.0)) 
    delta_z = fov_depth * abs(math.tan((fov_y_deg * math.pi / 180) / 2.0))

    # Define v0 and obst_pos
    p0 = np.array(agent_pos)
    obst_pos = np.array(obst_pos)
    ax.scatter(obst_pos[0], obst_pos[1], obst_pos[2], c='b', marker='o')
    ax.plot([agent_pos[0], obst_pos[0]], [agent_pos[1], obst_pos[1]], [agent_pos[2], obst_pos[2]], c='b', linestyle='--')

    # Compute the FOV vertices
    p0, p1, p2, p3, p4 = puma_specific_rotation(p0, delta_y, delta_z, fov_depth, agent_pos, agent_quat)

    ax.scatter(p0[0], p0[1], p0[2], c='g', marker='o')
    ax.scatter(p1[0], p1[1], p1[2], c='g', marker='o')
    ax.scatter(p2[0], p2[1], p2[2], c='g', marker='o')
    ax.scatter(p3[0], p3[1], p3[2], c='g', marker='o')
    ax.scatter(p4[0], p4[1], p4[2], c='g', marker='o')
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], c='g', linestyle='-')
    ax.plot([p0[0], p2[0]], [p0[1], p2[1]], [p0[2], p2[2]], c='g', linestyle='-')
    ax.plot([p0[0], p3[0]], [p0[1], p3[1]], [p0[2], p3[2]], c='g', linestyle='-')
    ax.plot([p0[0], p4[0]], [p0[1], p4[1]], [p0[2], p4[2]], c='g', linestyle='-')
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], c='g', linestyle='-')
    ax.plot([p2[0], p3[0]], [p2[1], p3[1]], [p2[2], p3[2]], c='g', linestyle='-')
    ax.plot([p3[0], p4[0]], [p3[1], p4[1]], [p3[2], p4[2]], c='g', linestyle='-')
    ax.plot([p4[0], p1[0]], [p4[1], p1[1]], [p4[2], p1[2]], c='g', linestyle='-')
    ax.view_init(30, 0)
    # plt.axis('equal')
    plt.show()
    # plt.savefig(f'/home/kota/tmp/fov_{str(t).replace(".","_")}.png')

if __name__ == "__main__":

    ##
    ## Define FOV-related parameters
    ##

    fov_x_deg = 76.0
    fov_y_deg = 47.0
    fov_depth = 6.0

    ##
    ## unit test
    ##

    # for i in range(100):
    # agent_pos = np.array([0.0, 0.0, 1.0])
    # agent_pos = np.array([0.002286666665913054, -0.0019351150260023175, 0.9989809339934611])
    # agent_quat = np.array([0.03217864911361818, 0.03575944747345984, 0.030381284631705568, 0.9983800749222427])
    # obst_pos = np.array([4.8604313353541215, 0.7090349694253053, -0.8144415744866915])

    agent_pos = np.array( [6.863542040943943, -0.034556468003291985, 1.076542315333])
    agent_quat = np.array( [-0.11677299963116804, -0.03364798857943447, 0.9909417643365874, 0.05715154516886451])
    obst_pos = np.array( [4.186282686798181, 0.047243829786385305, 2.9399056012080496])

    # agent_quat = quaternion_from_euler(random.uniform(-math.pi, math.pi), random.uniform(-math.pi, math.pi), random.uniform(-math.pi, math.pi))
    # agent_quat = np.array([0.0, 0.0, 0.0, 1.0])
    # obst_pos = np.array([random.uniform(-fov_depth, fov_depth), random.uniform(-fov_depth, fov_depth), random.uniform(-fov_depth, fov_depth)])
    # obst_pos = np.array([5.039, 0.56, -0.31])
    print(check_obst_is_in_FOV(agent_pos, agent_quat, obst_pos, fov_x_deg, fov_y_deg, fov_depth))
    visualization(agent_pos, agent_quat, obst_pos, fov_x_deg, fov_y_deg, fov_depth)
    time.sleep(0.1)
    



