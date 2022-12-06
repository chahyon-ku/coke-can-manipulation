import argparse
import cv2
import numpy as np
import pose.aruco_pose
import pose.soda_pose
import pose.se3
import all.state
from scipy.spatial.transform import Rotation as R


def add_arguments(parser):
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--model_type', type=str, default='')


def run(args, state: all.state.State):
    print(f'run_pose.run({state})')
    if state is None:
        pass
    else:
        templates = [np.array(cv2.imread('pose/templates/0.png', cv2.IMREAD_GRAYSCALE)),
                     np.array(cv2.imread('pose/templates/1.png', cv2.IMREAD_GRAYSCALE)),
                     np.array(cv2.imread('pose/templates/2.png', cv2.IMREAD_GRAYSCALE))]
        while True:
            color_image = state.get_color_image()
            depth_image = state.get_depth_image()
            intrinsic = state.get_intrinsic()
            aruco_t_camera = pose.aruco_pose.get_aruco_t_camera(color_image, intrinsic)
            aruco_t_soda = pose.soda_pose.get_aruco_t_soda(color_image, depth_image, intrinsic, aruco_t_camera, templates)
            state.set_aruco_t_camera(aruco_t_camera)
            state.set_aruco_t_soda(aruco_t_soda)

            camera_t_aruco = pose.se3.t_to_invt(aruco_t_camera)
            cv2.drawFrameAxes(color_image, intrinsic, .01, camera_t_aruco[3:],
                                             camera_t_aruco[:3], .1)

            camera_T_soda = pose.se3.t_to_T(camera_t_aruco) @ pose.se3.t_to_T(aruco_t_soda)
            camera_t_soda = pose.se3.T_to_t(camera_T_soda)
            cv2.drawFrameAxes(color_image, intrinsic, .01, camera_t_soda[3:], camera_t_soda[:3], .1)
            cv2.imshow('marker_image', color_image)
            cv2.waitKey(10)


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    run(args, None)


if __name__ == '__main__':
    main()
