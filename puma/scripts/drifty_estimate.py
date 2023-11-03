import numpy as np
from scipy.spatial.transform import Rotation as Rot

class DriftyEstimate():

    def __init__(self, position_drift, rotation_drift, position, orientation, dim=2):
        """
        Class for modeling drifty estimate

        Args:
            position_drift (np.array, shape=(dim,)): initial position offset
            rotation_drift (Rotation): Scipy Rotation object, initial rotation offset
            position (np.array, shape=(dim,)): initial true position
            orientation (Rotation): Scipy Rotation object, initial true orientation
        """
        self.dim = dim

        self.estimate = np.eye(self.dim+1)
        self.estimate[:self.dim,:self.dim] = orientation.as_matrix()[:self.dim,:self.dim]
        self.estimate[:self.dim,self.dim] = position[:self.dim]
        
        self.position = position
        self.orientation = orientation
        self.add_drift(position_drift, rotation_drift)
        
    def add_drift(self, position, rotation):
        """
        add drift to current estimate

        Args:
            position (np.array, shape=(dim,)): position drift
            rotation (Rotation): Scipy Rotation object representing drift in orientation
        """
        
        pos_drift_w = self.estimate[:self.dim,:self.dim] @ position.reshape((self.dim,1))
        self.estimate[:self.dim,self.dim] += pos_drift_w.reshape(-1)
        self.estimate[:self.dim,:self.dim] = self.estimate[:self.dim,:self.dim] @ rotation.as_matrix()[:self.dim,:self.dim]
        
    def update(self, position, orientation):
        """
        Updates drifty estimate

        Args:
            position (np.array, shape=(dim,)): Ground truth position
            orientation (Rotation): Scipy Rotation object, ground truth orientation

        Returns:
            np.array, shape=(dim,): Position estimate
            Rotation: Orientation estimate
        """
        T_WB = np.hstack([
            np.vstack([orientation.as_matrix()[:self.dim,:self.dim], np.zeros((1,self.dim))]),
            np.concatenate([np.array(position).reshape(-1), [1]]).reshape((self.dim+1,1))
        ])
        
        position_diff = position - self.position
        orientation_diff = self.orientation.as_matrix()[:self.dim,:self.dim].T @ orientation.as_matrix()[:self.dim,:self.dim]
        self.position = position
        self.orientation = orientation
        
        self.estimate[:self.dim,self.dim] += (self.estimate[:self.dim,:self.dim] @ T_WB[:self.dim,:self.dim].T @ position_diff[:self.dim].reshape((self.dim,1))).reshape(-1)
        self.estimate[:self.dim,:self.dim] = orientation_diff @ self.estimate[:self.dim,:self.dim]

        estimate_rot = np.eye(3)
        estimate_rot[:self.dim, :self.dim] = self.estimate[:self.dim,:self.dim].copy()

        return self.estimate[:self.dim,self.dim].copy(), Rot.from_matrix(estimate_rot)
            
    # @property
    # def T_drift(self):
    #     T_WB = np.hstack([
    #         np.vstack([self.orientation.as_matrix()[:self.dim,:self.dim], np.zeros((1,self.dim))]),
    #         np.concatenate([np.array(self.position).reshape(-1), [1]]).reshape((self.dim+1,1))
    #     ])
    #     T_drift_B = self.estimate
    #     return T_WB @ np.linalg.inv(T_drift_B)

    # give actual drift in world frame
    @property
    def drift(self):
        translational_drift = self.estimate[:self.dim,self.dim].copy() - self.position
        rotational_drift = self.orientation.as_matrix()[:self.dim,:self.dim].T @ self.estimate[:self.dim,:self.dim]
        tmp = np.eye(3)
        tmp[:self.dim,:self.dim] = rotational_drift
        return translational_drift, Rot.from_matrix(tmp).as_euler('xyz', degrees=True)