import argparse
import pybullet as pb
import all.state
import simulation.environment


def add_arguments(parser):
    parser.add_argument('--simulation_width', type=int, default=1280)
    parser.add_argument('--simulation_height', type=int, default=1080)


def run(args, state: all.state.State):
    env = simulation.environment.Environment()


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    run(args, None)


if __name__ == '__main__':
    main()
