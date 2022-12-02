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
    for i in range(6):
        env.update_can(i*0.05+0.4, i*0.05+0.7, np.pi/8*i)
    if args.random_pose:
        x = np.random.uniform(0.3, 0.7)
        y = np.random.uniform(0.7, 1.0)
        rz = np.random.uniform(-np.pi, np.pi)
        env.update_can(x, y, rz)
    else:
        env.update_can(args.can_x, args.can_y, args.can_rz)
    env.grasp_can()
    pb.disconnect()


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    run(args, None)


if __name__ == '__main__':
    main()
