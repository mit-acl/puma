# this file is from depth-estimation-experiments (https://gitlab.com/mit-acl/clipper/uncertain-localization/depth-estimation-experiments/)

import numpy as np
from box import Box
import yaml
import quaternion
from scipy.spatial.transform import Rotation as R

def readConfig(pathToConfigFile):
    with open(pathToConfigFile, "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))
    return cfg

def buildProjectionMatrixFromParams(fx, fy, cx, cy, s):
    K = np.eye(3)
    K[0,0] = fx
    K[1,1] = fy
    K[0,2] = cx
    K[1,2] = cy
    K[1,0] = s

    return K

def rotAndTransFromT(T):
    R = T[0:3,0:3]
    t = T[0:3,3]
    return (R,t)

def transfFromRotAndTransl(R,t):
    T = np.eye(4)
    T[0:3,0:3] = R
    T[0:3,3] = t
    return T

def compute_blob_mean_and_covariance(binary_image):

    # Create a grid of pixel coordinates.
    y, x = np.indices(binary_image.shape)

    # Threshold the binary image to isolate the blob.
    blob_pixels = (binary_image > 0).astype(int)

    # Compute the mean of pixel coordinates.
    mean_x, mean_y = np.mean(x[blob_pixels == 1]), np.mean(y[blob_pixels == 1])
    mean = (mean_x, mean_y)

    # Stack pixel coordinates to compute covariance using Scipy's cov function.
    pixel_coordinates = np.vstack((x[blob_pixels == 1], y[blob_pixels == 1]))

    # Compute the covariance matrix using Scipy's cov function.
    covariance_matrix = np.cov(pixel_coordinates)

    return mean, covariance_matrix

def plotErrorEllipse(ax,x,y,covariance,color=None,stdMultiplier=1,showMean=True,idText=None,marker='.'):

    covariance = np.asarray(covariance)

    (lambdas,eigenvectors) = np.linalg.eig(covariance)
    
    t = np.linspace(0,np.pi*2,30)
    
    lambda1 = lambdas[0]
    lambda2 = lambdas[1]
    
    scaledEigenvalue1 = stdMultiplier*np.sqrt(lambda1)*np.cos(t)
    scaledEigenvalue2 = stdMultiplier*np.sqrt(lambda2)*np.sin(t)
    
    scaledEigenvalues = np.vstack((scaledEigenvalue1,scaledEigenvalue2))
    
    ellipseBorderCoords = eigenvectors @ scaledEigenvalues
   
    ellipseBorderCoords_x = x+ellipseBorderCoords[0,:]
    ellipseBorderCoords_y = y+ellipseBorderCoords[1,:]
        
    if (color is not None):
        p = ax.plot(ellipseBorderCoords_x,ellipseBorderCoords_y,color=color)
    else:
        p = ax.plot(ellipseBorderCoords_x,ellipseBorderCoords_y)

    if (showMean):
        ax.plot(x,y,marker,color=p[0].get_color())

    if (idText is not None):
        ax.text(x,y,idText)

def to_homogeneous_coordinates(points):
    # Get the number of points and the dimension of the points
    num_points, dimension = points.shape

    # Create an array of ones with shape (numPoints, 1)
    ones_column = np.ones((num_points, 1), dtype=points.dtype)

    # Concatenate the ones_column to the right of the points array
    homogeneous_points = np.hstack((points, ones_column))

    return homogeneous_points


def from_homogeneous_coordinates(homogeneous_points, scale):
    regular_points = homogeneous_points[:, :-1]

    if (scale):
        scaling_factors = homogeneous_points[:, -1]
        regular_points = regular_points/scaling_factors[:, np.newaxis]

    return regular_points

def estimate_pixel_coordinates_from_pose_t2(points_t1, T1, T2, K):
    # Convert points_t1 to a Numpy array (Nx2 matrix)
    #points_t1 = np.array(points_t1)

    # Extract the rotation matrices from T1 and T2
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]

    # Compute the relative rotation between T1 and T2
    R_rel = np.matmul(R2, R1.T)

    # Invert the camera intrinsic matrix K
    K_inv = np.linalg.inv(K)

    # Convert points_t1 to homogeneous coordinates (add a column of ones)
    points_t1_homogeneous = np.hstack((points_t1, np.ones((points_t1.shape[0], 1))))

    # Create an empty array to store the estimated pixel coordinates in pose T2
    points_t2_estimated = np.empty_like(points_t1)

    # Iterate through each point observed from pose T1
    for i in range(points_t1.shape[0]):
        # Create a 3D point P1 in the camera coordinate system of pose T1
        P1 = np.matmul(K_inv, points_t1_homogeneous[i,:])

        print("P1=")
        print(P1)

        # Transform the 3D point P1 from pose T1 to pose T2
        P2 = np.matmul(R_rel, P1)

        # Project the 3D point P2 back to pixel coordinates in pose T2
        uv2_estimated = np.matmul(K, P2)
        u2_estimated, v2_estimated = uv2_estimated[:2] / uv2_estimated[2]

        # Store the estimated pixel coordinates
        points_t2_estimated[i] = [u2_estimated, v2_estimated]

    return points_t2_estimated

def compute_relative_rotation(T1, T2):
    # Extract the rotation matrices from T1 and T2 (upper-left 3x3 sub-matrices)
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]

    # Compute the relative rotation matrix R_rel
    R_rel = np.dot(R2, R1.T)

    return R_rel

def compute_relative_translation(T1, T2):
    # Extract the translation vectors from T1 and T2
    t1 = T1[:3, 3]
    t2 = T2[:3, 3]

    # Compute the relative translation vector t_rel
    t_rel = t2 - np.dot(T2[:3, :3], t1)

    return t_rel

def get_transformation_from_body_to_camera(camera):
    T_c_b = np.zeros((4,4))
    if camera == 'voxl':
        T_c_b[:3, :3] = get_rotation_matrix(0, -135, 90)
        camera_translation = np.array([0.09, 0.0, -0.03])
    elif camera == 't265_fisheye1':
        T_c_b[:3, :3] = get_rotation_matrix(0, 180, 90) #https://www.intelrealsense.com/wp-content/uploads/2019/09/Intel_RealSense_Tracking_Camera_Datasheet_Rev004_release.pdf
        camera_translation = np.array([-0.07, -0.03, -0.015])
    elif camera == 't265_fisheye2':
        T_c_b[:3, :3] = get_rotation_matrix(0, 180, 90) #https://www.intelrealsense.com/wp-content/uploads/2019/09/Intel_RealSense_Tracking_Camera_Datasheet_Rev004_release.pdf
        camera_translation = np.array([-0.07, 0.03, -0.015])
    elif camera == 'sim_camera':
        T_c_b[:3, :3] = get_rotation_matrix(0, 180, 90)
        camera_translation = np.array([0.0, 0.0, 0.0])
    else:
        raise ValueError(f"Invalid camera name in get_transformation_from_body_to_camera(): {camera}")
    
    T_c_b[:3, 3] = -np.matmul(T_c_b[:3, :3], camera_translation)
    T_c_b[3, 3] = 1.0
    
    return T_c_b

def compute_3d_position_of_each_centroid(mean_list, pose, camera, K):
    pos_3d = []
    for mean in mean_list:
        pos = compute_3d_position_of_centroid(mean, pose, camera, K)
        if pos is not None:
            pos_3d.append(pos)
    return pos_3d

def compute_3d_position_of_centroid(pixel, pose, camera, K):

    # get x_im from pixel
    x_im = np.array([pixel[0], pixel[1], 1.0])

    if K is None:
        # get camera intrinsics
        print("get K matrix - make sure K is updated in compute_3d_position_of_centroid()")
        if camera == 'voxl':
            # voxl camera calibration matrix (intrinsic parameters - look /data/modalai/opencv_tracking_intrinsics.yml)
            # K = np.array([[273.90235382142345, 0.0, 315.12271705027996], [0., 274.07405600616045, 241.28160498854680], [0.0, 0.0, 1.0]])
            # need to use a new K that is generated after undistortion
            K = np.array([[135.48433328, 0., 310.83524106], [0., 135.56926484, 241.39230258], [0., 0., 1.]])
        elif camera == 't265_fisheye1':
            K = np.array([[85.90185242, 0., 420.89700317], [0., 85.87307739, 404.22949219], [0., 0., 1.]])
        elif camera == 't265_fisheye2':
            K = np.array([[86.09328003, 0., 418.48440552], [0., 86.08716431, 400.93289185], [0., 0., 1.]])
        else:
            raise ValueError('Invalid camera name')

    # Transformatoin matrix T^b_w (from world to body)
    T_b_w = np.zeros((4,4))
    T_b_w[:3, :3] = np.linalg.inv(R.from_quat([pose[3], pose[4], pose[5], pose[6]]).as_matrix()) #from_quat() takes as input a quaternion in the form [x, y, z, w]
    T_b_w[:3, 3] = -np.matmul(T_b_w[:3,:3], np.array(pose[0:3]))
    T_b_w[3, 3] = 1.0

    # Transformation matrix T^c_b (from body to camera)
    T_c_b = get_transformation_from_body_to_camera(camera)

    # Transformation matrix T^c_w (from world to camera)
    T_c_w = np.matmul(T_c_b, T_b_w)

    # Get T and R from T_c_w
    R_c_w = T_c_w[:3, :3]
    t_c_w = T_c_w[:3, 3]

    # Get X_o from pose (or you can get it by - np.linalg.inv(R_c_w) @ t_c_w)
    # X_o = np.array(pose[0:3]) + camera_translation # which is the same as - np.linalg.inv(R_c_w) @ t_c_w
    X_o = - np.linalg.inv(R_c_w) @ t_c_w

    # compute lambda (depth) in X_w = X_o + lambda * (K * R_c_w)^-1 * x_im using flat earth assumption
    lambda_ = (0.0 - X_o[2]) / (np.matmul(np.linalg.inv(np.matmul(K, R_c_w)), x_im)[2])

    # compute X_w
    X_w = X_o + lambda_ * np.matmul(np.linalg.inv(np.matmul(K, R_c_w)), x_im)

    if lambda_ < 0:
        # raise ValueError(f"lambda_ {lambda_} < 0 \n pixel: {pixel}")
        print(f"lambda_ {lambda_} < 0 \n pixel: {pixel}")
        return 
    
    if abs(X_w[2]) > 1e-2:
        raise ValueError(f"X_w[2] {X_w[2]} is not 0")

    return [X_w[0], X_w[1]]

def get_rotation_matrix(roll, pitch, yaw):
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)

    R_r = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_p = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_y = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    R = np.matmul(R_y, np.matmul(R_p, R_r))
    return R

def quaternion_multiply(quaternion1, quaternion0): #https://stackoverflow.com/questions/39000758/how-to-multiply-two-quaternions-by-python-or-numpy
    w0 = quaternion0.w
    x0 = quaternion0.x
    y0 = quaternion0.y
    z0 = quaternion0.z
    w1 = quaternion1.w
    x1 = quaternion1.x
    y1 = quaternion1.y
    z1 = quaternion1.z
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def get_quaternion_from_euler(roll, pitch, yaw): #https://automaticaddison.com/how-to-convert-euler-angles-to-quaternions-using-python/
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return np.quaternion(qw, qx, qy, qz)

def quaternion_to_euler_angle_vectorized1(w, x, y, z): #https://stackoverflow.com/questions/56207448/efficient-quaternions-to-euler-transformation
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.where(t2>+1.0,+1.0,t2)
    #t2 = +1.0 if t2 > +1.0 else t2

    t2 = np.where(t2<-1.0, -1.0, t2)
    #t2 = -1.0 if t2 < -1.0 else t2
    Y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.arctan2(t3, t4)

    return X, Y, Z 

if __name__=="__main__":

    test_estimate_pixel_coordinates_from_pose_t2()