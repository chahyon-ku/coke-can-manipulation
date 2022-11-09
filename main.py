import argparse

import camera
import simulation
import pose

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    camera.add_arguments(parser)
    simulation.add_arguments(parser)
    pose.add_arguments(parser)
    args = parser.parse_args()

    camera.init(args)
    simulation.init(args)
    pose.init(args)

    while True:
        camera.step(args)
        simulation.step(args)
        pose.step(args)
