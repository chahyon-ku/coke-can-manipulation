import argparse
import pybullet as pb
import all.state
import simulation.environment
import simulation.pybullet_utils as pbu
import numpy as np

def add_arguments(parser):
    parser.add_argument('--generate_data', type=bool, default=False)
    parser.add_argument('--num_data', type=int, default=10)
    parser.add_argument('--can_x', type=float, default=0.6)
    parser.add_argument('--can_y', type=float, default=0.7)
    parser.add_argument('--can_rz', type=float, default=0.0)
    parser.add_argument('--random_pose', type=bool, default=True)


def run(args, state: all.state.State):
    env = simulation.environment.Environment()
    i = 0
    prev_x = 0
    prev_y = 0
    while True:
        aruco_t_soda = state.get_aruco_t_soda()
        in2m = 0.0254
        x_aruco = 2.25*in2m + 0.05
        y_aruco = in2m + 0.05
        x_table = 15*in2m + 0.5
        y_table = 36*in2m + 0.47
        x_world = aruco_t_soda[0] + x_table - x_aruco
        y_world = aruco_t_soda[1] + y_table - y_aruco
        env.update_can(x_world, y_world, 0)
        print(i, prev_x, prev_y)
        if (aruco_t_soda[0] - prev_x) ** 2 + (aruco_t_soda[1] - prev_y) ** 2 < 1e-1:
            i += 1
        else:
            i = 0
        prev_x = aruco_t_soda[0]
        prev_y = aruco_t_soda[1]

        if state.get_is_grasping():
            break

    env.grasp_can()
    pb.disconnect()


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    run(args, None)


if __name__ == '__main__':
    main()
