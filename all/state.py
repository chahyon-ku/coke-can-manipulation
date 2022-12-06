import numpy as np


class State(object):
    def __init__(self):
        self.color_image = np.zeros((540, 960, 3), dtype=np.uint8)
        self.depth_image = np.zeros((540, 960), dtype=np.uint16)
        self.intrinsic = np.zeros((3, 3))
        self.aruco_t_camera = np.zeros(6)
        self.aruco_t_soda = np.zeros(6)
        self.is_grasping = False

    def set_color_image(self, color_image):
        self.color_image = color_image

    def get_color_image(self):
        return np.copy(self.color_image)

    def set_depth_image(self, depth_image):
        self.depth_image = depth_image

    def get_depth_image(self):
        return self.depth_image

    def set_intrinsic(self, intrinsic):
        self.intrinsic = intrinsic

    def get_intrinsic(self):
        return self.intrinsic

    def set_aruco_t_camera(self, aruco_t_camera):
        self.aruco_t_camera = aruco_t_camera

    def get_aruco_t_camera(self):
        return self.aruco_t_camera

    def set_aruco_t_soda(self, aruco_t_soda):
        self.aruco_t_soda = aruco_t_soda

    def get_aruco_t_soda(self):
        return self.aruco_t_soda

    def set_is_grasping(self, is_grasping):
        self.is_grasping = is_grasping

    def get_is_grasping(self):
        return self.is_grasping
