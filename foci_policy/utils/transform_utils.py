"""
Coordinate transformation utilities for pose representation
"""
import torch
import numpy as np
import scipy.spatial.transform
import open3d as o3d


class Rotation(scipy.spatial.transform.Rotation):
    @classmethod
    def identity(cls):
        return cls.from_quat([0.0, 0.0, 0.0, 1.0])


class Transform(object):
    """Rigid spatial transform between coordinate systems in 3D space.
    Attributes:
        rotation (scipy.spatial.transform.Rotation)
        translation (np.ndarray)
    """

    def __init__(self, rotation, translation):
        assert isinstance(rotation, scipy.spatial.transform.Rotation)
        assert isinstance(translation, (np.ndarray, list))

        self.rotation = rotation
        self.translation = np.asarray(translation, np.double)

    def as_matrix(self):
        """Represent as a 4x4 matrix."""
        return np.vstack(
            (np.c_[self.rotation.as_matrix(), self.translation], [0.0, 0.0, 0.0, 1.0])
        )

    def to_dict(self):
        """Serialize Transform object into a dictionary."""
        return {
            "rotation": self.rotation.as_quat().tolist(),
            "translation": self.translation.tolist(),
        }

    def to_list(self):
        return np.r_[self.rotation.as_quat(), self.translation]

    def __mul__(self, other):
        """Compose this transform with another."""
        rotation = self.rotation * other.rotation
        translation = self.rotation.apply(other.translation) + self.translation
        return self.__class__(rotation, translation)

    def transform_point(self, point):
        return self.rotation.apply(point) + self.translation

    def transform_vector(self, vector):
        return self.rotation.apply(vector)

    def inverse(self):
        """Compute the inverse of this transform."""
        rotation = self.rotation.inv()
        translation = -rotation.apply(self.translation)
        return self.__class__(rotation, translation)

    @classmethod
    def from_matrix(cls, m):
        """Initialize from a 4x4 matrix."""
        rotation = Rotation.from_matrix(m[:3, :3])
        translation = m[:3, 3]
        return cls(rotation, translation)

    @classmethod
    def from_dict(cls, dictionary):
        rotation = Rotation.from_quat(dictionary["rotation"])
        translation = np.asarray(dictionary["translation"])
        return cls(rotation, translation)

    @classmethod
    def from_list(cls, list):
        rotation = Rotation.from_quat(list[:4])
        translation = list[4:]
        return cls(rotation, translation)

    @classmethod
    def identity(cls):
        """Initialize with the identity transformation."""
        rotation = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
        translation = np.array([0.0, 0.0, 0.0])
        return cls(rotation, translation)

    @classmethod
    def look_at(cls, eye, center, up):
        """Initialize with a LookAt matrix.
        Returns:
            T_eye_ref, the transform from camera to the reference frame, w.r.t.
            which the input arguments were defined.
        """
        eye = np.asarray(eye)
        center = np.asarray(center)

        forward = center - eye
        forward /= np.linalg.norm(forward)

        right = np.cross(forward, up)
        right /= np.linalg.norm(right)

        up = np.asarray(up) / np.linalg.norm(up)
        up = np.cross(right, forward)

        m = np.eye(4, 4)
        m[:3, 0] = right
        m[:3, 1] = -up
        m[:3, 2] = forward
        m[:3, 3] = eye

        return cls.from_matrix(m).inverse()


def to_array(tensor):
    """
    Conver tensor to array
    """
    if not isinstance(tensor, np.ndarray):
        if tensor.device == torch.device('cpu'):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor


def to_o3d_pcd(xyz, colors=None, grey=False, red=False, orange=False, normals=None):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pts = to_array(xyz)
    pcd.points = o3d.utility.Vector3dVector(pts)
    if normals is not None:
        normals = to_array(normals)
        # pcd.colors = o3d.utility.Vector3dVector(np.array([colors]*pts.shape[0]))
        pcd.normals = o3d.utility.Vector3dVector(to_array(normals))
    if colors is not None:
        # pcd.colors = o3d.utility.Vector3dVector(np.array([colors]*pts.shape[0]))
        pcd.colors = o3d.utility.Vector3dVector(to_array(colors))
    if grey:
        pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(pts)+0.1)
    if red:
        c = np.zeros_like(pts)
        c[:,0] = 1.0
        pcd.colors = o3d.utility.Vector3dVector(c)
    if orange:
        c = np.zeros_like(pts)
        c[:,:] = np.asarray([255, 165, 0])/255
        pcd.colors = o3d.utility.Vector3dVector(c)
    return pcd



def rotation_matrix_to_6d(rotation_matrix):
    """
    Convert 3x3 rotation matrix to 6D representation
    Takes the first two columns of the rotation matrix
    
    Args:
        rotation_matrix: (..., 3, 3) rotation matrix
    Returns:
        rot_6d: (..., 6) 6D rotation representation [col1_x, col1_y, col1_z, col2_x, col2_y, col2_z]
    """
    # Extract first two columns
    col1 = rotation_matrix[..., :, 0]  # (..., 3) - first column
    col2 = rotation_matrix[..., :, 1]  # (..., 3) - second column
    
    # Concatenate to form 6D representation
    rot_6d = torch.cat([col1, col2], dim=-1)  # (..., 6)
    return rot_6d


def rotation_6d_to_matrix(rot_6d):
    """
    Convert 6D rotation representation to 3x3 rotation matrix
    Uses Gram-Schmidt orthogonalization to ensure valid rotation matrix
    
    Args:
        rot_6d: (..., 6) tensor representing first two columns of rotation matrix
    Returns:
        rot_matrix: (..., 3, 3) rotation matrix
    """
    batch_shape = rot_6d.shape[:-1]
    rot_6d = rot_6d.reshape(-1, 6)
    
    # Extract first two columns
    col1 = rot_6d[:, :3]  # (B, 3)
    col2 = rot_6d[:, 3:]  # (B, 3)
    
    # Normalize first column
    col1 = col1 / torch.norm(col1, dim=-1, keepdim=True)
    
    # Gram-Schmidt orthogonalization for second column
    col2 = col2 - (col1 * col2).sum(dim=-1, keepdim=True) * col1
    col2 = col2 / torch.norm(col2, dim=-1, keepdim=True)
    
    # Cross product for third column
    col3 = torch.cross(col1, col2, dim=-1)
    
    # Stack columns to form rotation matrix
    rot_matrix = torch.stack([col1, col2, col3], dim=-1)  # (B, 3, 3)
    
    # Reshape back to original batch shape
    rot_matrix = rot_matrix.reshape(*batch_shape, 3, 3)
    return rot_matrix


def matrix_to_pose_9d(pose_matrix):
    """
    Convert 4x4 pose matrix to 9D representation (3 pos + 6D rot)
    
    Args:
        pose_matrix: (..., 4, 4) transformation matrix
    Returns:
        pose_9d: (..., 9) pose in 9D format [position(3), rotation_6d(6)]
    """
    # Extract position
    position = pose_matrix[..., :3, 3]  # (..., 3)
    
    # Extract rotation and convert to 6D
    rotation_matrix = pose_matrix[..., :3, :3]  # (..., 3, 3)
    rotation_6d = rotation_matrix_to_6d(rotation_matrix)  # (..., 6)
    
    # Concatenate position and rotation
    pose_9d = torch.cat([position, rotation_6d], dim=-1)  # (..., 9)
    return pose_9d


def pose_9d_to_matrix(pose_9d):
    """
    Convert 9D pose (3 pos + 6D rot) to 4x4 transformation matrix
    
    Args:
        pose_9d: (..., 9) pose in 9D format [position(3), rotation_6d(6)]
    Returns:
        pose_matrix: (..., 4, 4) transformation matrix
    """
    batch_shape = pose_9d.shape[:-1]
    
    # Extract position and rotation
    position = pose_9d[..., :3]  # (..., 3)
    rotation_6d = pose_9d[..., 3:]  # (..., 6)
    
    # Convert rotation to matrix
    rot_matrix = rotation_6d_to_matrix(rotation_6d)  # (..., 3, 3)
    
    # Build 4x4 transformation matrix
    pose_matrix = torch.zeros(*batch_shape, 4, 4, device=pose_9d.device, dtype=pose_9d.dtype)
    pose_matrix[..., :3, :3] = rot_matrix
    pose_matrix[..., :3, 3] = position
    pose_matrix[..., 3, 3] = 1.0
    
    return pose_matrix

def create_identity_pose_9d(batch_shape, device='cpu'):
    """
    Create identity pose in 9D representation
    
    Args:
        batch_shape: tuple, shape of batch dimensions
        device: device to create tensor on
    Returns:
        identity_pose: (..., 9) identity pose
    """
    pose_9d = torch.zeros(*batch_shape, 9, device=device)
    # Set rotation to identity: first column [1, 0, 0], second column [0, 1, 0]
    pose_9d[..., 3] = 1.0  # First column, first element
    pose_9d[..., 7] = 1.0  # Second column, second element
    return pose_9d