import argparse
import threading
import numpy as np
import run_camera
import run_pose
import run_simulation


def add_arguments(parser):
    run_camera.add_arguments(parser)
    run_pose.add_arguments(parser)
    run_simulation.add_arguments(parser)


def run(args, state):
    camera_thread = threading.Thread(target=run_camera.run, name='camera_thread', args=(args, state))
    pose_thread = threading.Thread(target=run_pose.run, name='pose_thread', args=(args, state))
    simulation_thread = threading.Thread(target=run_simulation.run, name='simulation_thread', args=(args, state))
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
