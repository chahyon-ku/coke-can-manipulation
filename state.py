class State():
    def __init__(self):
        self.image = np.zeros((1, 1))
        self.intrinsic = np.zeros((3, 3))
        self.camera_t_aruco = np.zeros((4, 4))
        self.camera_t_soda = np.zeros((4, 4))

    def set_image(self, image):
        self.image = image

    def get_image(self):
        return self.image

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