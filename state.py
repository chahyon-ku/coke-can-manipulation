import numpy as np


class State:
    def __init__(self):
        self.color_image = np.zeros((3, 540, 960))
        self.depth_image = np.zeros((3, 540, 960))
        self.intrinsic = np.zeros((3, 3))
        self.camera_t_aruco = np.zeros((4, 4))
        self.camera_t_soda = np.zeros((4, 4))

    def set_color_image(self, color_image):
        self.color_image = color_image

    def get_color_image(self):
        return self.color_image

    def set_depth_image(self, depth_image):
        self.depth_image = depth_image

    def get_depth_image(self):
        return self.depth_image

    def set_intrinsic(self, intrinsic):
        self.intrinsic = intrinsic

    def get_intrinsic(self):
        return self.intrinsic

    def set_camera_t_aruco(self, camera_t_aruco):
        self.camera_t_aruco = camera_t_aruco

    def get_camera_t_aruco(self):
        return self.camera_t_aruco

    def set_camera_t_soda(self, camera_t_soda):
        self.camera_t_soda = camera_t_soda

    def get_camera_t_soda(self):
        return self.camera_t_soda