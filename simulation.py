import argparse
import pybullet as pb


def add_arguments(parser):
    parser.add_argument('--simulation_width', type=int, default=1980)
    parser.add_argument('--simulation_height', type=int, default=1280)


def run(args, state):
    physics_client = pb.connect(pb.GUI, options=f"--width={args.simulation_width}--height={args.simulation_height}")
    while True:
        pass


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    run(args, None)


if __name__ == '__main__':
    main()
