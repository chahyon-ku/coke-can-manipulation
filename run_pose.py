import argparse

import pose.aruco_pose
import pose.soda_pose
import all.state


def add_arguments(parser):
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--model_type', type=str, default='')


def run(args, state: all.state.State):
    print(f'run_pose.run({state})')
    if state is None:
        pass
    else:
        while True:
            color_image = state.get_color_image()
            depth_image = state.get_depth_image()
            intrinsic = state.get_intrinsic()
            aruco_t_camera = pose.aruco_pose.get_aruco_t_camera(color_image, intrinsic)
            aruco_t_soda = pose.soda_pose.get_aruco_t_soda(color_image, depth_image, intrinsic, aruco_t_camera)
            state.set_aruco_t_camera(aruco_t_camera)
            state.set_aruco_t_soda(aruco_t_soda)


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    run(args, None)


if __name__ == '__main__':
    main()
