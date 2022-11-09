import argparse

import pybullet as pb


def step(args):
    pass


def init(args):
    physics = pb.connect(pb.GUI, options=f"--width={args.width} --height={args.height}")
    pb.setGravity(0, 0, -9.81)
    pb.setRealTimeSimulation(True)


def add_arguments(parser):
    parser.add_argument('--simulation_width', type=int, default=1980)
    parser.add_argument('--simulation_height', type=int, default=1280)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    init(args)
    while True:
        step(args)
