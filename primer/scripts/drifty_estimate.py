import numpy as np
from scipy.spatial.transform import Rotation as Rot

class DriftyEstimate():

    def __init__(self, position_drift, rotation_drift, position, orientation):
        """
        Class for modeling drifty estimate

        Args:
            position_drift (np.array, shape=(3,)): initial position offset
            rotation_drift (Rotation): Scipy Rotation object, initial rotation offset
            position (np.array, shape=(3,)): initial true position
            orientation (Rotation): Scipy Rotation object, initial true orientation
        """
        self.estimate = np.eye(4)
        self.estimate[:3,:3] = orientation.as_matrix()[:3,:3]
        self.estimate[:3,3] = position[:3]
        
        self.position = position
        self.orientation = orientation
        self.add_drift(position_drift, rotation_drift)
        
    def add_drift(self, position, rotation):
        """
        add drift to current estimate

        Args:
            position (np.array, shape=(3,)): position drift
            rotation (Rotation): Scipy Rotation object representing drift in orientation
        """
        pos_drift_w = self.estimate[:3,:3] @ position.reshape((3,1))
        self.estimate[:3,3] += pos_drift_w.reshape(-1)
        self.estimate[:3,:3] = self.estimate[:3,:3] @ rotation.as_matrix()
        
    def update(self, position, orientation):
        """
        Updates drifty estimate

        Args:
            position (np.array, shape=(3,)): Ground truth position
            orientation (Rotation): Scipy Rotation object, ground truth orientation

        Returns:
            np.array, shape=(3,): Position estimate
            Rotation: Orientation estimate
        """
        T_WB = np.hstack([
            np.vstack([orientation.as_matrix(), np.zeros((1,3))]),
            np.concatenate([np.array(position).reshape(-1), [1]]).reshape((4,1))
        ])
        
        position_diff = position - self.position
        orientation_diff = self.orientation.as_matrix().T @ orientation.as_matrix()
        self.position = position
        self.orientation = orientation
        
        self.estimate[:3,3] += (self.estimate[:3,:3] @ T_WB[:3,:3].T @ position_diff[:3].reshape((3,1))).reshape(-1)
        self.estimate[:3,:3] = orientation_diff @ self.estimate[:3,:3]

        return self.estimate[:3,3].copy(), Rot.from_matrix(self.estimate[:3,:3])
            
    @property
    def T_drift(self):
        T_WB = np.hstack([
            np.vstack([self.orientation.as_matrix(), np.zeros((1,3))]),
            np.concatenate([np.array(self.position).reshape(-1), [1]]).reshape((4,1))
        ])
        T_drift_B = self.estimate
        return T_WB @ np.linalg.inv(T_drift_B)