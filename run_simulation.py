import argparse
import pybullet as pb
import all.state
import simulation.environment
import simulation.pybullet_utils as pbu


def add_arguments(parser):
    parser.add_argument('--generate_data', type=bool, default=False)
    parser.add_argument('--num_data', type=int, default=10)
    parser.add_argument('--can_x', type=float, default=0.6)
    parser.add_argument('--can_y', type=float, default=0.7)


def run(args, state: all.state.State):
    env = simulation.environment.Environment()
    for i in range(10000):
        pbu.step_real(1)
    env.grasp_can(args.can_x, args.can_y)
    # pbu.step_real(1)
    pb.disconnect()


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    run(args, None)


if __name__ == '__main__':
    main()
