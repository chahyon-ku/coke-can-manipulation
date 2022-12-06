import numpy as np
from scipy.spatial.transform import Rotation as R


def t_to_T(t):
    T = np.zeros((4, 4))
    T[:3, :3] = R.from_euler('XYZ', t[3:]).as_matrix()
    T[:3, 3] = t[:3]
    T[3, 3] = 1
    return T


def t_to_invt(t):
    invt = T_to_t(T_to_invT(t_to_T(t)))
    return invt


def T_to_invT(T):
    invT = np.zeros((4, 4))
    invT[:3, :3] = np.transpose(T[:3, :3])
    invT[:3, 3] = -invT[:3, :3] @ T[:3, 3]
    invT[3, 3] = 1
    return invT


def T_to_t(T):
    t = np.zeros(6)
    t[:3] = T[:3, 3]
    t[3:] = R.from_matrix(T[:3, :3]).as_euler('XYZ')
    return t
