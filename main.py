import argparse
import threading

import numpy as np

import camera
import pose
import simulation


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


def add_arguments(parser):
    camera.add_arguments(parser)
    pose.add_arguments(parser)
    simulation.add_arguments(parser)


def run(args, state):
    camera_thread = threading.Thread(target=camera.run, name='camera_thread', args=(args, state))
    pose_thread = threading.Thread(target=pose.run, name='pose_thread', args=(args, state))
    simulation_thread = threading.Thread(target=simulation.run, name='simulation_thread', args=(args, state))
    camera_thread.run()
    pose_thread.run()
    simulation_thread.run()


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    state = None
    run(args, state)


if __name__ == '__main__':
    main()
